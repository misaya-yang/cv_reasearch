"""Remove confirmed redundant download material; retain canonical research assets."""
import hashlib
import json
from pathlib import Path
import zipfile
import zlib

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/takeover_20261001_v1/storage_cleanup_20261001.json'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    if OUT.exists():raise FileExistsError(OUT)
    report={'status':'RUNNING','user_authorized_cleanup':True,'deletions':[],'retained_canonical_weights':[],'skipped':[]}
    def save():OUT.write_text(json.dumps(report,indent=2)+'\n')
    def remove(path,reason,proof):
        if not path.exists():return
        row={'path':str(path),'bytes':path.stat().st_size,'reason':reason,'proof':proof}
        save();path.unlink();report['deletions'].append(row);save();print(json.dumps(row),flush=True)
    save()
    weights=ROOT/'assets/checkpoints'
    # A completed checkpoint has already loaded in this research run.
    canonical=weights/'sam_vit_b_01ec64.pth';partial=weights/'sam_vit_b_01ec64.pth.partial'
    if partial.exists() and canonical.stat().st_size>partial.stat().st_size:
        with partial.open('rb') as x,canonical.open('rb') as y:
            prefix_equal=True
            for b in iter(lambda:x.read(8<<20),b''):
                if y.read(len(b))!=b:prefix_equal=False;break
        remove(partial,'abandoned incomplete download; complete canonical checkpoint retained',{'canonical':str(canonical),'canonical_bytes':canonical.stat().st_size,'prefix_matches':prefix_equal,'complete_checkpoint_previously_loaded':True})
    for p in weights.glob('*speed_probe.bin'):
        remove(p,'obsolete speed probe, not a model weight',{})
    probe=weights/'hq_cdn_speed_probe.bin'
    if probe.exists():remove(probe,'obsolete speed probe, not a model weight',{})
    archive=weights/'efficient_sam_vits.pt.zip';extracted=weights/'efficient_sam_vits.pt'
    if archive.exists() and extracted.exists():
        with zipfile.ZipFile(archive) as z:
            match=next(i for i in z.infolist() if i.filename=='efficient_sam_vits.pt');h=hashlib.sha256()
            with z.open(match) as f:
                for b in iter(lambda:f.read(8<<20),b''):h.update(b)
        if h.hexdigest()==sha(extracted):remove(archive,'archive duplicates verified extracted canonical model',{'canonical':str(extracted),'member_sha256':h.hexdigest()})
    # Physical duplicate frozen checkpoint paths remain readable via hard links.
    buckets={}
    for p in sorted(weights.iterdir()):
        if p.suffix not in ('.pt','.pth') or not p.is_file():continue
        key=(p.stat().st_size,sha(p))
        if key not in buckets:buckets[key]=p;report['retained_canonical_weights'].append(str(p));continue
        other=buckets[key]
        if other.stat().st_ino!=p.stat().st_ino:
            size=p.stat().st_size;tmp=p.with_name(p.name+'.dedup.tmp');tmp.hardlink_to(other);tmp.replace(p)
            report['deletions'].append({'path':str(p),'bytes':size,'reason':'duplicate frozen weight bytes replaced by canonical hard link','proof':{'canonical':str(other),'sha256':key[1]}});save()
    archives=ROOT/'assets/datasets/archives'
    davis=archives/'DAVIS-2017-trainval-480p.zip'
    # First verify completed assembly before deleting all redundant range parts.
    for folder in sorted((archives/'.chunks').glob('*')):
        if not folder.is_dir():continue
        final=archives/folder.name;parts=sorted(folder.glob('part_*.bin'))
        if not final.is_file() or not parts:continue
        if sum(p.stat().st_size for p in parts)!=final.stat().st_size:report['skipped'].append(str(folder));continue
        h=hashlib.sha256()
        for p in parts:
            with p.open('rb') as f:
                for b in iter(lambda:f.read(8<<20),b''):h.update(b)
        if h.hexdigest()==sha(final):
            for p in parts:remove(p,'redundant verified archive assembly range',{'completed_archive':str(final),'assembled_sha256':h.hexdigest()})
    if davis.exists():
        verified=0;complete=True
        with zipfile.ZipFile(davis) as z:
            for info in z.infolist():
                if info.is_dir():continue
                target=ROOT/'assets/datasets'/info.filename
                if not target.is_file() or target.stat().st_size!=info.file_size:complete=False;break
                crc=0
                with target.open('rb') as f:
                    for b in iter(lambda:f.read(8<<20),b''):crc=zlib.crc32(b,crc)
                if crc!=info.CRC:complete=False;break
                verified+=1
        if complete:remove(davis,'complete archive duplicates CRC-verified extracted DAVIS data',{'verified_files':verified,'extracted_root':str(ROOT/'assets/datasets/DAVIS')})
        else:report['skipped'].append({'archive':str(davis),'reason':'extraction not fully verified; archive retained'})
    report.update(status='COMPLETED',freed_bytes=sum(r['bytes'] for r in report['deletions']));save();print(json.dumps({'status':'COMPLETED','freed_bytes':report['freed_bytes']}),flush=True)


if __name__=='__main__':main()
