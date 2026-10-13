"""Metadata-only original-photo exclusion ledger and fresh catalog inventory.

No sampling budget, final task manifest, labels, model, or metrics are produced.
The root runner can filter existing official loaders by the exported photo IDs.
Prepared seed manifests are identity bridges, not evidence of model exposure.
"""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import pickle
import re

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / 'cv_data'
OUT = DATA / 'a/x1013_photo_holdout'
DOMAINS = ('coco','lvis','paco_part','pascal_part','suim','isaid','isic','lung','fundus','deepglobe_road')


def read(path):
    return json.loads(Path(path).read_text())


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def canonical(path, domain=''):
    path = str(path).replace('\\', '/')
    name = Path(path).name
    if 'ISIC_' in name:
        return 'isic:' + re.search(r'ISIC_\d+', name)[0]
    if re.search(r'(?:COCO_|/coco/|COCO2014)', path, re.I):
        match = re.search(r'(\d{12})\.(?:jpg|png)$', name, re.I)
        if match:
            return 'coco:' + match[1]
    if 'Pascal-Part' in path or domain == 'pascal_part':
        match = re.search(r'(\d{4}_\d{6})', name)
        if match:
            return 'pascal:' + match[1]
    if 'iSAID' in path or domain == 'isaid':
        match = re.match(r'(P\d+)', name)
        if match:
            return 'isaid:' + match[1]
    if 'LungSegmentation' in path or domain == 'lung':
        match = re.search(r'((?:CHN|MCU)CXR_\d+_\d+)', name)
        if match:
            return 'lung:' + match[1]
    if 'Fundus/' in path:
        match = re.search(r'Fundus/(train|test)/Original/([^/]+)', path)
        if match:
            return 'fundus:' + match[1] + '/' + match[2]
    if 'SUIM/images/' in path:
        return 'suim:' + Path(name).stem
    if domain == 'deepglobe_road' or 'deepglobe' in path.lower():
        match = re.search(r'(\d+)_sat\.(?:jpg|png)$', name)
        if match:
            return 'road:' + match[1]
    return None


class Ledger:
    def __init__(self):
        self.parent, self.known = {}, {}
        self.exposed = set()
        self.unresolved = defaultdict(set)

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def add(self, path, domain='', rgb=None, file_sha=None, expose=False, source=None):
        if not path or Path(str(path)).suffix.lower() not in ('.jpg','.jpeg','.png','.bmp'):
            return False
        path = str(path)
        if path.startswith(('ref/','query/','support/')):
            # A pack-relative filename is never globally unique across packs.
            match = re.search(r'/rcg2_group_?([^/]+)/', source or '')
            pack = match[1] if match else ('suim' if '/previous_ablation_smoke/suim/' in (source or '') else None)
            if pack and (DATA/'episodes/claude_packs'/pack/'data'/path).is_file():
                path = str(DATA/'episodes/claude_packs'/pack/'data'/path)
            else:
                path = str(Path(source or 'unknown').parent/'unresolved_pack_relative'/path)
        if path.startswith(str(DATA)+'/'):
            path = path[len(str(DATA))+1:]
        aliases = ['path:' + path]
        photo = canonical(path, domain)
        if photo:
            aliases.append('photo:' + photo)
        if rgb:
            aliases.append('rgb:' + str(rgb))
        if file_sha:
            aliases.append('file:' + str(file_sha))
        for alias in aliases:
            a, b = self.find(aliases[0]), self.find(alias)
            if a != b:
                self.parent[max(a,b)] = min(a,b)
        if photo:
            self.known['photo:' + photo] = photo
        if expose:
            self.exposed.add(aliases[0])
            if not photo:
                self.unresolved[(domain or 'unknown', source or '')].add(aliases[0])
        return True

    def consume(self, value, expose, source, domain=''):
        count = 0
        if isinstance(value, list):
            return sum(self.consume(x, expose, source, domain) for x in value)
        if not isinstance(value, dict):
            return 0
        domain = value.get('dataset', domain)
        if isinstance(domain, dict):
            domain = ''
        domain = {'PACO':'paco_part','PASCAL':'pascal_part','LVIS':'lvis','COCO':'coco','SUIM':'suim'}.get(domain,domain)
        if not domain:
            domain = next((d for d in ('paco_part','pascal_part','lvis','suim','coco') if d in source.lower()),'')
        for role in ('reference','query','support','target'):
            path = value.get(role+'_path')
            if not path and isinstance(value.get(role), str):
                path = value[role]
            original = value.get('source_reference' if role in ('reference','support') else 'source_query')
            if original and '/outputs/claude_packs/' in str(original):
                path = 'episodes/claude_packs/' + str(original).split('/outputs/claude_packs/', 1)[1]
            if path:
                count += self.add(path, domain, value.get(role+'_rgb_hash'),
                    value.get('source_files', {}).get(path), expose, source)
        if isinstance(value.get('view'), dict):
            view = value['view']
            path = view.get('path')
            count += self.add(path, domain, view.get('rgb_hash'),
                value.get('source_files', {}).get(path), expose, source)
        # Old SUIM exporter has a verified role-to-original-image map.
        if value.get('image_path') and value.get('image_sha256'):
            count += self.add(value['image_path'], domain or 'suim', None,
                              value['image_sha256'], expose, source)
        for key, child in value.items():
            if key not in ('source_files','fit','raw','raw_binding','raw_feature_metadata','metrics','iu','source_sha256'):
                if isinstance(child,(dict,list)):
                    count += self.consume(child, expose, source, domain)
        return count


def sources():
    registry = DATA/'a/x1012s2_confirm1000/exclusion_sources.json'
    paths = {Path(p) for p in read(registry)}
    paths.update(DATA/'a'/p for p in ('x1012_P0/observations.json','x1012s2_confirm1000/manifest.json',
                 'x1012s2_confirm1000/manifest.baselines.json','x1013_photo_holdout/legacy_rgb_bridge.json'))
    retired = DATA/'repository_retired/20261013T014935Z/evidence/local'
    for pattern in ('*manifest*.json','*bindings*.json','pairs.json','episodes.jsonl','rows.json','exposed*.json'):
        paths.update(retired.rglob(pattern))
    paths.update((DATA/'repository_assets/20261013T014935Z/outputs/m4').glob('development*protocol*/manifest.json'))
    paths.update(DATA/p for p in ('cache/astra_granularity_20261008/coco_fresh600/rows.json',
                 'cache/confirm1200_conditional_v1/inputs/manifest.json','setup/suim_pack_audit/old_pack_identity_map.json'))
    return sorted(p for p in paths if p.is_file())


def prepared_only(path):
    text = str(path)
    return any(x in text for x in ('/official_s0_manifest_', '/fresh_manifests_',
        '/lvis_official600_partitions_', '/new_dataset600_preparation_',
        '/extended_benchmarks_seed0_', '/chest_seed0_manifest_', '/fundus_seed0_manifest_',
        '/prepared/','/source_extension/','/sources/'))


def catalog(ledger):
    result, metadata = [], []
    def add(d, fold, cls, path, source, file_sha=None, name=None):
        path = Path(path)
        if not path.is_file():
            return
        rel = str(path.relative_to(DATA))
        ledger.add(rel, d, file_sha=file_sha)
        result.append(dict(dataset=d, fold=fold, class_id=cls, path=rel,
                           canonical_id=canonical(rel,d), loader_name=name or path.name,
                           catalog_source=str(source)))
    def js(path):
        metadata.append(path); return read(path)
    def pk(path):
        metadata.append(path)
        with path.open('rb') as stream: return pickle.load(stream)
    coco = DATA/'demo4_cache/data/COCO2014'
    for fold in range(4):
        p=coco/f'splits/val/fold{fold}.pkl'
        for cls, names in pk(p).items():
            if int(cls)%4 == fold:
                for name in names: add('coco',fold,int(cls),coco/name,p,name=name)
    p=DATA/'datasets/ics/LVIS/lvis_val.pkl';val=pk(p)
    cats=[c for c,v in val.items() if len(v)>1]
    for fold in range(10):
        for pos in range(len(cats)//10):
            cls=cats[fold+10*pos]
            for name in val[cls]:add('lvis',fold,cls,DATA/'datasets/ics/LVIS/coco'/name,p,name=name)
    p=DATA/'datasets/ics/PACO-Part/paco/paco_part_train.pkl';train=pk(p)
    p=DATA/'datasets/ics/PACO-Part/paco/paco_part_val.pkl';val=pk(p)
    for pos,cls in enumerate(train['cid2img']):
        names={str(next(iter(x))):str(next(iter(x.values()))) for x in val['cid2img'].get(cls,[])}
        if len(names)<2:continue
        for name in names.values():
            name='/'.join(name.split('/')[-2:])
            add('paco_part',pos%4,cls,DATA/'datasets/ics/PACO-Part/coco'/name,p,name=name)
    p=DATA/'datasets/ics/Pascal-Part/VOCdevkit/VOC2010/all_obj_part_to_image.json'
    val=js(p)
    for fold,group in enumerate(('animals','indoor','person','vehicles')):
        for obj,record in val[group]['object'].items():
            for part,pool in record['part'].items():
                cls=obj+'+'+part
                if cls=='aeroplane+TAIL' or not pool['train'] or not pool['val']:continue
                for name in pool['val']:add('pascal_part',fold,cls,p.parent/'JPEGImages'/f'{name}.jpg',p,name=name)
    p=DATA/'setup/manifests/ics_datasets.json';files=js(p)['files'];shas={x['path']:x['sha256'] for x in files}
    for x in files:
        parts=x['path'].split('/')
        if len(parts)==4 and parts[:2]==['SUIM','masks']:
            name=Path(parts[-1]).stem;rgb='SUIM/images/'+name+'.jpg'
            add('suim',-1,parts[2],DATA/'datasets/ics'/rgb,p,shas.get(rgb),name)
    for fold in range(3):
        p=DATA/f'datasets/ics/iSAID/splits/val/fold{fold}.txt';metadata.append(p)
        for line in p.read_text().splitlines():
            if not line:continue
            name,cls=line.split('__')
            add('isaid',fold,int(cls)-1,DATA/'datasets/ics/iSAID/val/images'/f'{name}.png',p,name=name)
    p=DATA/'setup/asset_downloads/isic_input/isic_prepare_receipt.json'
    for x in js(p)['files']:
        add('isic',-1,int(x['fold'])-1,DATA/x['split_image'],p,x['split_image_sha256'],x['image_id'])
        ledger.add(x['raw_image'],'isic',file_sha=x['raw_image_sha256'])
    for d,folder in [('lung','Chest'),('fundus','Fundus')]:
        p=DATA/f'setup/asset_downloads/{folder}_autonomous_20261010/preparation_receipt.json'
        files=js(p)['files'];by_path={x['path']:x for x in files}
        for x in files:
            path=x['path']
            if d=='lung' and '/masks/' in path and path.endswith('.png'):
                name=Path(path).name;name=name if 'MCUCXR' in name else name[:-9]+'.png'
                rgb='datasets/ics/LungSegmentation/CXR_png/'+name
            elif d=='fundus' and '/test/Original/' in path and path.endswith('.png'):
                rgb=path;name=Path(path).name
            else:continue
            add(d,-1,0,DATA/rgb,p,by_path.get(rgb,{}).get('sha256'),name)
    p=DATA/'a/deepglobe200_preparation_20261009/asset_inventory_final.jsonl';metadata.append(p)
    for line in p.read_text().splitlines():
        x=json.loads(line)
        if x['source_path'].endswith('_sat.jpg'):
            add('deepglobe_road',-1,0,p.parent/x['path'],p,x['sha256'],x['source_path'])
    return result,metadata


def main():
    ledger=Ledger();source_records=[];seen=set()
    for path in sources():
        payload=path.read_bytes();digest=hashlib.sha256(payload).hexdigest()
        expose=not prepared_only(path)
        # Equal content with a real-run source must still mark its photos exposed.
        key=(digest,expose)
        if key in seen:
            source_records.append(dict(path=str(path),sha256=digest,mode='duplicate_content',exposure=expose));continue
        seen.add(key)
        try:
            value=([json.loads(l) for l in payload.decode().splitlines() if l.strip()]
                   if path.suffix=='.jsonl' else json.loads(payload))
            declared_rows = len(value) if isinstance(value,list) else None
            # The Chest run stopped at 167/600; its full prepared roster is not exposure.
            if path.parent.name == 'chest_foris_seed0_20261010' and isinstance(value,list):
                completed={int(p.stem) for p in (path.parent/'records').glob('*.json') if p.stem.isdigit()}
                value=[row for i,row in enumerate(value) if i in completed]
            n=ledger.consume(value,expose,str(path))
            source_records.append(dict(path=str(path),sha256=digest,mode='used_experiment_manifest' if expose else 'prepared_identity_bridge_only',role_records=n,declared_rows=declared_rows,consumed_rows=len(value) if isinstance(value,list) else None))
        except (ValueError,UnicodeDecodeError) as exc:
            source_records.append(dict(path=str(path),sha256=digest,mode='unparsed',error=str(exc)))
    # Apply the existing verified SUIM old-export identity map, including resized PNGs.
    for row in read(DATA/'setup/suim_pack_audit/old_pack_identity_map.json')['mapping']:
        for role,sub in [('support','ref'),('query','query')]:
            old='path:'+f"episodes/claude_packs/suim/data/{sub}/{row['pack_row']:05d}.png"
            new='path:'+row[role]['image_path']
            a,b=ledger.find(old),ledger.find(new)
            if a!=b:ledger.parent[max(a,b)]=min(a,b)
    items,catalog_sources=catalog(ledger)
    exposed={ledger.find(x) for x in ledger.exposed}
    named=defaultdict(set)
    for alias,photo in ledger.known.items():named[ledger.find(alias)].add(photo)
    canonical_by_root={root:sorted(ids)[0] for root,ids in named.items()}
    unresolved=[]
    for (domain,source),aliases in ledger.unresolved.items():
        unknown={ledger.find(a) for a in aliases if ledger.find(a) not in named}
        if unknown:unresolved.append(dict(dataset_hint=domain,source=source,unknown_parent_groups=len(unknown),examples=sorted(unknown)[:5]))
    fresh=[];pool=defaultdict(set);total=defaultdict(set);seen_items=set()
    for item in items:
        identity=ledger.find('path:'+item['path']);canonical_id=canonical_by_root.get(identity,item['canonical_id'])
        item['canonical_id']=canonical_id
        total[item['dataset']].add(canonical_id)
        if identity not in exposed:
            key=(item['dataset'],item['fold'],str(item['class_id']),item['path'])
            if key in seen_items:continue
            seen_items.add(key);fresh.append(item)
            pool[(item['dataset'],item['fold'],str(item['class_id']))].add(canonical_id)
    summary={}
    for d in DOMAINS:
        groups={f'{f}/{c}':sorted(ids) for (domain,f,c),ids in pool.items() if domain==d}
        photos={i for ids in groups.values() for i in ids}
        usable={i for ids in groups.values() if len(ids)>=2 for i in ids}
        relevant = [u for u in unresolved if u['dataset_hint'] in (d,'unknown') or
                    d in ('coco','lvis','paco_part') and u['dataset_hint'] in ('coco','lvis','paco_part')]
        summary[d]=dict(local_catalog_original_photos=len(total[d]),remaining_after_identified_exposure=len(photos),
                        new_photos_in_a_class_with_two_or_more=len(usable),same_class_pair_groups=sum(len(v)>=2 for v in groups.values()),
                        class_photo_counts={c:len(ids) for c,ids in groups.items()},
                        unresolved_relevant_sources=len(relevant),
                        freshness_status='needs_unresolved_legacy_parent_review' if relevant else 'metadata_exposure_registry_disjoint')
    write('exposure_sources.json',source_records)
    write('exposed_photos.json',dict(photo_ids=sorted({p for r in exposed for p in named.get(r,())}),
         alias_groups=[sorted(named[r]) for r in sorted(exposed) if r in named and len(named[r])>1],unresolved_legacy=unresolved))
    write('fresh_catalog.json',fresh)
    write('inventory.json',dict(scope='Metadata-only availability, no final budget or tasks chosen',
         per_domain=summary,source_files=len(source_records),unresolved_legacy_sources=unresolved,
         catalog_sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in dict.fromkeys(catalog_sources)],
         rules=['Original photo aliases merge across crops/classes/roles/domains; iSAID uses parent scene.',
                'Prepared full seed manifests only provide identity bridges and do not mark exposure.',
                'Potential remaining counts are not a freshness guarantee while legacy parent identities remain unresolved.',
                'Root selects final budget and filters existing loaders; no query GT pixels, scores or encoders read here.']))
    for d,s in summary.items():print(d,s['local_catalog_original_photos'],s['remaining_after_identified_exposure'],s['same_class_pair_groups'])
    print('unresolved legacy sources',len(unresolved),'output',OUT)


if __name__=='__main__':main()
