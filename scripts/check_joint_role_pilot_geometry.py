"""Focused numerical checks for the new R1-G geometry, without real labels."""
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ics.methods import joint_reference_bp as bp
from ics.methods import joint_role_pilot_geometry as model


def main(output):
    checks={}
    q=np.array([[.25,.25],[.75,.25],[.25,.75],[.75,.75]])/64
    graph=bp.build_grid_graph(q,(2,2));valid=np.ones((4,1),bool)
    point=np.full((4,1,2),.5/64)
    for scale in (.125,1.,8.):
        value=model.pair_distance(point,valid,np.tile(scale*np.eye(2),(4,1,1)),graph)
        checks['point_collapse_cost_'+str(scale)]=bool(np.allclose(value,1,atol=1e-14))
    alias=q[:,None,:]
    checks['same_coarse_atom_fine_positions_can_fit']=bool(np.allclose(
        model.pair_distance(alias,valid,np.tile(np.eye(2),(4,1,1)),graph),0,atol=1e-14))
    mask=np.zeros((1024,1024),bool);mask[:16,:16]=1;mask[4:12,4:12]=0
    support=model.ReferenceSupport(mask,(1024,1024))
    centroid=np.array([[[8/1024,8/1024]]])
    projected=support.project(centroid,np.ones((1,1),int),np.ones((1,1),bool))
    checks['nonconvex_support_projects_illegal_centroid']=bool(np.allclose(projected,[[[8/1024,4/1024]]]))
    checks['support_projection_idempotent']=bool(np.array_equal(projected,
        support.project(projected,np.ones((1,1),int),np.ones((1,1),bool))))
    yy,xx=np.indices((128,128));xy=np.stack(((xx+.5)/128,(yy+.5)/128),-1).reshape(-1,2)
    full,colour,info=model.make_graph(xy,(128,128),(1024,1024),(1024,1024))
    checks['sampling_derived_span46']=info['span_x']==info['span_y']==46
    checks['all_graph_edges_cross_colours']=bool(np.all(colour[full.u]!=colour[full.v]))
    checks['four_neighbour_and_span_count']=len(full.u)==53504
    # Both labels have identical appearance and symmetric legal support.
    mask=(np.indices((1024,1024)).sum(0)%2)==0
    observation=dict(b0=np.full((4,2),.5),valid=np.ones((4,2),bool),
        role=np.tile([0,1],(4,1)),refpatch=np.zeros((4,2),int),
        ref_xy=np.full((4,2,2),.5),query_xy=q*64)
    symmetric=model.solve(observation,mask,(1024,1024),(1024,1024),grid_hw=(2,2))
    checks['ambiguous_symmetric_roles_stay_half']=bool(np.allclose(symmetric.p_fg,.5,atol=1e-10))
    checks['fixed_belief_MM_decreases']=symmetric.diagnostics['MM']['fixed_belief_MM_nonincrease']
    # Two foreground instances with a background bridge; no global capacity.
    mask=np.zeros((1024,1024),bool);mask[:,:512]=1
    fg=np.array([True,True,False,False,True,True,False,False])
    pfg=np.where(fg,.9,.1)
    repeated=dict(b0=np.stack((1-pfg,pfg),1),valid=np.ones((8,2),bool),
        role=np.tile([0,1],(8,1)),refpatch=np.tile([32*64+40,32*64+8],(8,1)),
        ref_xy=np.tile([[40.5,32.5],[8.5,32.5]],(8,1,1)),
        query_xy=np.column_stack(((np.arange(8)+.5)*8,np.full(8,32.))))
    result=model.solve(repeated,mask,(1024,1024),(1024,1024),grid_hw=(1,8))
    checks['two_instances_and_background_bridge_preserved']=bool(np.array_equal(result.p_fg>.5,fg))
    legal=model.ReferenceSupport(mask,(1024,1024)).project(result.positions,
        repeated['refpatch']*2+repeated['role'],repeated['valid'])
    checks['MM_positions_remain_in_reference_roles']=bool(np.allclose(legal,result.positions,atol=1e-14))
    checks['finite_multi_instance_posterior']=bool(np.isfinite(result.p_fg).all())
    assert all(checks.values()),checks
    report=dict(state='PASSED',checks=checks,graph=info,
        symmetric_p_fg=symmetric.p_fg.tolist(),two_instance_p_fg=result.p_fg.tolist(),
        synthetic_only=True,model_calls=0,real_query_GT_reads=0,
        limits='Functional and implementation checks; not accuracy evidence. Strong-prior correction bound remains.')
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main(sys.argv[1])
