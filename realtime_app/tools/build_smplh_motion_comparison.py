"""Build a frame-synchronized inspection page for the saved stereo2D comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PAGE = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SMPL-H · 三组448帧对照</title>
<style>*{box-sizing:border-box}body{margin:0;color:#000;background:#fff;font:14px system-ui,sans-serif}
header{padding:12px 18px;border-bottom:1px solid #000}h1{font-size:20px;margin:0 0 10px}
.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap}select,input,button{background:#fff;color:#000;border:1px solid #000;padding:5px}
#frame{width:65px}#seek{flex:1;min-width:160px}p{line-height:1.5;margin:8px 0}#viewer{display:block;width:100%;height:calc(100vh - 200px);min-height:660px;border:0}
a{color:#000}#summary{font-variant-numeric:tabular-nums}</style>
<header><h1>SMPL-H · 同源双目二维跟踪 · 三组448帧对照</h1>
<div class="controls"><label>结果 <select id="variant" aria-label="结果"></select></label>
<label>帧 <input id="frame" type="number" min="0" max="447" value="0" aria-label="对照帧"></label>
<input id="seek" type="range" min="0" max="447" value="0" aria-label="对照进度"><span id="time"></span>
<a id="standalone" target="_blank">独立打开当前结果</a></div>
<p id="summary"></p><p>切换保留帧与观察视角。三组固定同一主线beta、同源图像及本轮场景。黑色实线保留全部有限三角化点；误差只统计，真正缺失不补点。
地面与助步器是Stage条件显示参考，不能据此认定独立世界运动。两组MAP尚未收敛，速度/加速度默认留空，可在下方勾选原始估计。</p></header>
<iframe id="viewer" title="当前结果的完整三维与双目检查"></iframe>
<script>
const config=__CONFIG__;
const select=document.getElementById('variant'),iframe=document.getElementById('viewer');
let frame=0,view=null;
for(const variant of config.variants){const option=document.createElement('option');option.value=variant.name;option.textContent=variant.label;select.append(option);}
const urlFrame=Number(new URLSearchParams(location.search).get('frame')||0);
if(Number.isInteger(urlFrame)&&urlFrame>=0&&urlFrame<config.times.length) frame=urlFrame;
function syncControls(){
  document.getElementById('frame').value=frame;document.getElementById('seek').value=frame;
  document.getElementById('time').textContent=`${frame} / ${config.times.length-1} · ${(config.times[frame]-config.times[0]).toFixed(3)} s`;
  const variant=config.variants.find(item=>item.name===select.value);
  document.getElementById('standalone').href=variant.page+'?frame='+frame;
  window.comparisonAudit={variant:variant.name,frame,view};
}
function sendFrame(){iframe.contentWindow.postMessage({type:'smplh-set-frame',frame,view},location.origin);}
function switchVariant(){
  const variant=config.variants.find(item=>item.name===select.value);
  document.getElementById('summary').textContent=variant.summary;
  iframe.src=variant.page+'?frame='+frame;syncControls();
}
function setFrame(value){const next=Number(value);if(!Number.isFinite(next))return;frame=Math.max(0,Math.min(config.times.length-1,Math.round(next)));syncControls();sendFrame();}
document.getElementById('frame').oninput=event=>setFrame(event.target.value);
document.getElementById('seek').oninput=event=>setFrame(event.target.value);
select.onchange=switchVariant;
window.addEventListener('message',event=>{
  if(event.source!==iframe.contentWindow||event.origin!==location.origin)return;
  if(event.data?.type==='smplh-ready'){sendFrame();return;}
  if(event.data?.type!=='smplh-frame-changed')return;
  const next=event.data.frame;if(!Number.isInteger(next)||next<0||next>=config.times.length)return;
  frame=next;view=event.data.view;syncControls();
});
switchVariant();
</script></html>'''


def build(root: Path):
    comparison = json.loads((root / 'comparison.json').read_text(encoding='utf-8'))['comparison']
    variants = []
    reference = None
    labels = {'ekf_sigma15': 'EKF · 整体VPoser ×1',
              'map_sigma15': '迭代MAP · 整体VPoser ×1',
              'map_sigma075': '迭代MAP · 整体VPoser ×4'}
    for name, label in labels.items():
        run = root / name
        metadata = json.loads((run / 'view/metadata.json').read_text(encoding='utf-8'))
        with np.load(run / 'result.npz') as result:
            identity = [result[key].copy() for key in ('pair_id', 'timestamp_s', 'betas')]
        if reference is None:
            reference = identity
        elif any(not np.array_equal(a, b) for a, b in zip(reference, identity)):
            raise ValueError('comparison requires identical frame identities, timestamps and fixed beta')
        if metadata['frames'] != 448 or metadata['vertices'] != 6890 or metadata['faces'] != 13776:
            raise ValueError('expected all448 saved complete SMPL-H meshes')
        stats = comparison[name]
        variants.append({'name': name, 'label': label, 'page': f'../{name}/view/body_browser.html',
                         'summary': f"{label}；双目原始回投中位 {stats['raw_error_px']['median']:.2f} px / P95 {stats['raw_error_px']['p95']:.2f} px；"
                                    f"≤25px {stats['fit_le25px_rate_whole10752']:.2%}；可发布关节导数 {stats['derivative_available_samples']}/5376。"})
    config = {'variants': variants, 'times': reference[1].tolist()}
    output = root / 'visualization/index.html'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(PAGE.replace('__CONFIG__', json.dumps(config, ensure_ascii=False)), encoding='utf-8')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    print(build(parser.parse_args().root))
