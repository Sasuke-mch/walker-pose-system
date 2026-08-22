from __future__ import annotations
import argparse, csv, json
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--annotation', required=True)
    p.add_argument('--predictions', required=True)
    p.add_argument('--summary-json', required=True)
    p.add_argument('--summary-csv', required=True)
    p.add_argument('--per-image-csv', required=True)
    p.add_argument('--oks-thresholds', nargs='+', type=float, default=[0.5,0.75])
    p.add_argument('--category-name', default='person')
    return p.parse_args()

def div(a,b):
    return float(a/b) if b else 0.0

def write_csv(path, rows):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        if not rows: return
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def main():
    a=parse_args()
    preds=json.loads(Path(a.predictions).read_text(encoding='utf-8-sig'))
    if not isinstance(preds,list) or not preds:
        raise ValueError('Prediction JSON must be a non-empty list.')
    gt=COCO(a.annotation)
    img_ids=sorted(map(int,gt.getImgIds())); img_set=set(img_ids)
    bad=sorted({int(x['image_id']) for x in preds if int(x['image_id']) not in img_set})
    if bad: raise ValueError(f'Unknown image ids: {bad[:10]}')
    cat_ids=gt.getCatIds(catNms=[a.category_name])
    if len(cat_ids)!=1: raise ValueError(f'Expected one category named {a.category_name}, got {cat_ids}')
    cat_id=int(cat_ids[0])
    preds=[x for x in preds if int(x.get('category_id',cat_id))==cat_id]
    counts=Counter(int(x['image_id']) for x in preds)
    max_per=max(counts.values(), default=0); max_dets=max(1,max_per)
    dt=gt.loadRes(preds)
    ev=COCOeval(gt,dt,'keypoints')
    ev.params.imgIds=img_ids; ev.params.catIds=[cat_id]
    ev.params.iouThrs=np.asarray(a.oks_thresholds,dtype=np.float64)
    ev.params.maxDets=[max_dets]
    ev.params.areaRng=[[0.0,1.0e10]]; ev.params.areaRngLbl=['all']
    ev.evaluate()
    by_img={int(e['image_id']):e for e in ev.evalImgs if e is not None}
    thrs=[float(x) for x in a.oks_thresholds]
    totals={t:dict(tp=0,fp=0,fn=0,ignored_predictions=0,ignored_gt=0) for t in thrs}
    rows=[]
    for img_id in img_ids:
        info=gt.imgs[img_id]; e=by_img.get(img_id)
        row={'image_id':img_id,'file_name':info.get('file_name',''),
             'predictions_post_oks_nms':counts.get(img_id,0),
             'over_coco_maxdets_20':int(counts.get(img_id,0)>20),
             'predictions_beyond_20':max(0,counts.get(img_id,0)-20)}
        if e is None:
            ann_ids=gt.getAnnIds(imgIds=[img_id],catIds=[cat_id])
            g=len(gt.loadAnns(ann_ids))
            for t in thrs:
                lab=f'{t:.2f}'.replace('.','')
                row.update({f'gt_{lab}':g,f'tp_{lab}':0,f'fp_{lab}':counts.get(img_id,0),f'fn_{lab}':g,
                            f'precision_{lab}':0.0,f'recall_{lab}':0.0,f'f1_{lab}':0.0})
                totals[t]['fp']+=counts.get(img_id,0); totals[t]['fn']+=g
            rows.append(row); continue
        dm=np.asarray(e['dtMatches']); gm=np.asarray(e['gtMatches'])
        di=np.asarray(e['dtIgnore'],dtype=bool); gi=np.asarray(e['gtIgnore'],dtype=bool)
        for i,t in enumerate(thrs):
            vdt=~di[i]; vgt=~gi; mdt=dm[i]>0; mgt=gm[i]>0
            tp=int(np.count_nonzero(mdt & vdt)); fp=int(np.count_nonzero((~mdt)&vdt))
            fn=int(np.count_nonzero((~mgt)&vgt)); ignp=int(np.count_nonzero(~vdt)); igng=int(np.count_nonzero(~vgt))
            p=div(tp,tp+fp); r=div(tp,tp+fn); f1=div(2*p*r,p+r)
            lab=f'{t:.2f}'.replace('.','')
            row.update({f'gt_{lab}':tp+fn,f'tp_{lab}':tp,f'fp_{lab}':fp,f'fn_{lab}':fn,
                        f'precision_{lab}':p,f'recall_{lab}':r,f'f1_{lab}':f1})
            totals[t]['tp']+=tp; totals[t]['fp']+=fp; totals[t]['fn']+=fn
            totals[t]['ignored_predictions']+=ignp; totals[t]['ignored_gt']+=igng
        rows.append(row)
    over20=sum(c>20 for c in counts.values())
    beyond20=sum(max(0,c-20) for c in counts.values())
    summary=[]; metrics={}
    for t in thrs:
        v=totals[t]; tp=v['tp']; fp=v['fp']; fn=v['fn']
        p=div(tp,tp+fp); r=div(tp,tp+fn); f1=div(2*p*r,p+r)
        row={'oks_threshold':t,'images':len(img_ids),'gt_instances':tp+fn,
             'predictions_post_oks_nms':len(preds),'tp':tp,'fp':fp,'fn':fn,
             'precision':p,'recall':r,'f1':f1,'fp_per_image':div(fp,len(img_ids)),
             'fn_per_image':div(fn,len(img_ids)),'ignored_predictions':v['ignored_predictions'],
             'ignored_gt':v['ignored_gt'],'max_dets_used':max_dets,
             'images_over_20_predictions':over20,'predictions_beyond_20':beyond20,
             'max_predictions_per_image':max_per}
        summary.append(row); metrics[f'oks_{t:.2f}']=row
    out={'annotation':a.annotation,'predictions':a.predictions,
         'evaluation_scope':'All post-OKS-NMS predictions; no per-image maxDets=20 truncation.',
         'images':len(img_ids),'predictions_post_oks_nms':len(preds),
         'mean_predictions_per_image':div(len(preds),len(img_ids)),
         'max_predictions_per_image':max_per,'images_over_20_predictions':over20,
         'predictions_beyond_20':beyond20,'max_dets_used':max_dets,'metrics':metrics}
    Path(a.summary_json).parent.mkdir(parents=True,exist_ok=True)
    Path(a.summary_json).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    write_csv(a.summary_csv,summary); write_csv(a.per_image_csv,rows)
    print('===== ALL-PREDICTION POSE EVALUATION =====')
    print(f'Images: {len(img_ids)}')
    print(f'Post-OKS-NMS predictions: {len(preds)}')
    print(f'maxDets used: {max_dets} (covers every prediction)')
    print(f'Images over 20 predictions: {over20}')
    print(f'Predictions beyond standard maxDets=20: {beyond20}')
    for row in summary:
        print(f"OKS={row['oks_threshold']:.2f} | TP={row['tp']} FP={row['fp']} FN={row['fn']} | "
              f"P={row['precision']:.6f} R={row['recall']:.6f} F1={row['f1']:.6f} | "
              f"FP/image={row['fp_per_image']:.6f}")
    print(f"Summary JSON: {a.summary_json}")
    print(f"Summary CSV: {a.summary_csv}")
    print(f"Per-image CSV: {a.per_image_csv}")

if __name__=='__main__':
    main()
