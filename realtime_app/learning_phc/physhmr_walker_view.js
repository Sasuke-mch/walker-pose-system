'use strict';
const rawEdges=[[0,1],[0,2],[1,3],[2,4],[0,5],[0,6],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
let previousInput=-1;
function drawWalkerScene(){
  if(!data.walker)return;
  for(const [a,b] of data.walker_edges)line(project(...data.walker[frame][a]),project(...data.walker[frame][b]),'#555e56',4);
  const cameras=data.cameras[frame].map(p=>project(...p));line(cameras[0],cameras[1],'#697d91',2);
  ctx.fillStyle='#697d91';for(let i=0;i<2;i++){const p=cameras[i];ctx.fillRect(p[0]-4,p[1]-4,8,8);ctx.fillText(i?'右相机':'左相机',p[0]+6,p[1]);}
  const raw=data.raw_points[frame];
  for(const [a,b] of rawEdges){if(raw[a].every(v=>v!==null)&&raw[b].every(v=>v!==null))line(project(...raw[a]),project(...raw[b]),'#202020',1);}
  for(let j=0;j<17;j++){if(!data.raw_accepted[frame][j]&&raw[j].every(v=>v!==null)){const p=project(...raw[j]);line([p[0]-3,p[1]-3],[p[0]+3,p[1]+3],'#d08428');line([p[0]-3,p[1]+3],[p[0]+3,p[1]-3],'#d08428');}}
  // Mark the actual author simulation plane, without moving it to the scene ground.
  const z=data.simulation_plane_z_m,x=center[0],y=center[1];
  line(project(x-.7,y-.7,z),project(x+.7,y-.7,z),'#948294',1);
  const label=project(x+.7,y-.7,z);ctx.fillStyle='#7e667e';ctx.fillText(`仿真平面 Z=${z.toFixed(3)}m`,label[0],label[1]-4);
}
function drawWalkerInputs(){
  if(!data.images||previousInput===frame)return;previousInput=frame;
  const selected=frame;
  for(let side=0;side<2;side++){
    const c=document.getElementById('input'+side),g=c.getContext('2d');
    g.clearRect(0,0,c.width,c.height);g.fillStyle='#444';g.fillText(`加载帧 ${selected}…`,8,20);c.dataset.frame='loading';
    const image=new Image();
    image.onload=()=>{
      if(frame!==selected)return;
      g.drawImage(image,0,0,c.width,c.height);c.dataset.frame=selected;
      function point(p){if(p[0]===null||p[1]===null)return null;return side?[(1079-p[1])*.2,p[0]*.2]:[p[1]*.2,(1919-p[0])*.2];}
      for(let j=5;j<17;j++){
        const observed=point(data.observed_pixels[selected][side][j]),predicted=point(data.predicted_pixels[selected][side][j]);
        if(observed){g.strokeStyle='#ed972b';g.lineWidth=1.5;g.beginPath();g.arc(...observed,3,0,Math.PI*2);g.stroke();}
        if(predicted){g.fillStyle='#1bbd8e';g.beginPath();g.arc(...predicted,2,0,Math.PI*2);g.fill();}
      }
    };
    image.onerror=()=>{const g=document.getElementById('input'+side).getContext('2d');g.clearRect(0,0,216,384);g.fillText('原图加载失败',10,20);};
    image.src=encodeURI(data.images[selected][side]);
  }
  const value=data.image_median_px[frame];
  document.getElementById('sceneinfo').textContent=`采集 t=${data.source_times[frame].toFixed(3)}s / 仿真 t=${(frame*data.dt).toFixed(3)}s；Stage=${data.stage[frame]}。原图有限回投中位差 ${value===null?'不可用':value.toFixed(1)+'px'}。黑线：有限三角化参考；橙叉：严格门拒绝。${frame===0?'首帧是初始化状态。':''}`;
  document.getElementById('details').textContent=JSON.stringify({reasons:data.raw_reasons[frame],scene:Object.fromEntries(Object.entries(data.scene_status).map(([k,v])=>[k,v[frame]]))},null,2);
}
