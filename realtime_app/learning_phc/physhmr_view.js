'use strict';
const $=id=>document.getElementById(id), canvas=$('scene'), ctx=canvas.getContext('2d');
let data, mesh, reference, position, faces, frame=0, playing=false, last=0;
let yaw=.65, pitch=.18, zoom=1, panX=0, panY=0, center, width, height, drag;
const unpack=(str,Type=Float32Array)=>{const s=atob(str),b=new Uint8Array(s.length);for(let i=0;i<s.length;i++)b[i]=s.charCodeAt(i);return new Type(b.buffer);};
function project(x,y,z){
  // Fixed projection center only: no frame-dependent translation or rescaling.
  x-=center[0]; y=-(y-center[1]); z-=.85;
  const u=Math.cos(yaw)*x-Math.sin(yaw)*y, d=Math.sin(yaw)*x+Math.cos(yaw)*y;
  const v=Math.cos(pitch)*z-Math.sin(pitch)*d, depth=Math.sin(pitch)*z+Math.cos(pitch)*d;
  const scale=Math.min(width/3.4,height/2.5)*zoom;
  return [width/2+u*scale+panX,height/2-v*scale+panY,depth];
}
function line(a,b,color,weight=1){ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.strokeStyle=color;ctx.lineWidth=weight;ctx.stroke();}
function grid(){
  const x0=Math.floor(center[0]*2)/2, y0=Math.floor(center[1]*2)/2;
  ctx.fillStyle='#edf0e9';ctx.beginPath();
  [[x0-2,y0-2,0],[x0+2,y0-2,0],[x0+2,y0+2,0],[x0-2,y0+2,0]].forEach((p,i)=>{p=project(...p);i?ctx.lineTo(...p.slice(0,2)):ctx.moveTo(...p.slice(0,2));});ctx.closePath();ctx.fill();
  for(let i=-4;i<=4;i++){
    line(project(x0+i*.5,y0-2,0),project(x0+i*.5,y0+2,0),'#cdd5ca');
    line(project(x0-2,y0+i*.5,0),project(x0+2,y0+i*.5,0),'#cdd5ca');
  }
  const o=[x0-1,y0-1,0],p=project(...o);
  [[.5,0,0,'X / 0.5m'],[0,.5,0,'Y / 0.5m'],[0,0,.5,'Z / 0.5m']].forEach(a=>{const q=project(o[0]+a[0],o[1]+a[1],a[2]);line(p,q,'#46554d',2);ctx.fillStyle='#46554d';ctx.font='12px system-ui';ctx.fillText(a[3],q[0]+5,q[1]-5);});
}
function skeleton(array,color,weight){
  const points=[];for(let j=0;j<24;j++){const k=frame*72+j*3;points.push(project(array[k],array[k+1],array[k+2]));}
  for(let j=1;j<24;j++)line(points[j],points[data.parents[j]],color,weight);
  ctx.fillStyle=color;for(const p of points){ctx.beginPath();ctx.arc(p[0],p[1],3,0,Math.PI*2);ctx.fill();}
}
function surface(){
  const base=frame*6890*3, points=new Array(6890);
  for(let i=0;i<6890;i++){const k=base+i*3;points[i]=project(mesh[k],mesh[k+1],mesh[k+2]);}
  const triangles=new Array(faces.length/3);
  for(let i=0;i<triangles.length;i++){const a=points[faces[3*i]],b=points[faces[3*i+1]],c=points[faces[3*i+2]];triangles[i]=[i,(a[2]+b[2]+c[2])/3];}
  triangles.sort((a,b)=>b[1]-a[1]);
  ctx.strokeStyle='rgba(20,100,79,.12)';ctx.lineWidth=.35;
  for(const [i] of triangles){
    const ia=faces[i*3],ib=faces[i*3+1],ic=faces[i*3+2],a=points[ia],b=points[ib],c=points[ic];
    const k=base+ia*3,l=base+ib*3,m=base+ic*3;
    const u=[mesh[l]-mesh[k],mesh[l+1]-mesh[k+1],mesh[l+2]-mesh[k+2]],v=[mesh[m]-mesh[k],mesh[m+1]-mesh[k+1],mesh[m+2]-mesh[k+2]];
    const n=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]],len=Math.hypot(...n)||1;
    const shade=.55+.45*Math.abs((n[0]*.35+n[1]*-.2+n[2]*.915)/len);
    ctx.fillStyle=`rgb(${Math.round(38*shade)},${Math.round(160*shade)},${Math.round(127*shade)})`;
    ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.lineTo(c[0],c[1]);ctx.closePath();ctx.fill();ctx.stroke();
  }
}
function render(){
  if(!data)return;
  ctx.clearRect(0,0,width,height);ctx.fillStyle='#f5f5f2';ctx.fillRect(0,0,width,height);grid();
  if($('physics').checked)surface();
  if($('joints').checked)skeleton(position,'#164337',1.5);
  // Deliberately visible through the surface, to inspect local disagreement.
  if($('reference').checked)skeleton(reference,'#d47b2d',2.4);
  $('seek').value=frame;$('frame').value=frame;
  const k=frame*72;
  $('metrics').textContent=`第 ${frame+1} / ${data.frames} 步 · t=${((frame+1)*data.dt).toFixed(3)}s　全局参考差 ${data.global_error_mm[frame].toFixed(1)} mm　去根平移后参考差 ${data.relative_error_mm[frame].toFixed(1)} mm　骨盆高度 ${position[k+2].toFixed(3)} m　作者终止标志：${data.termination[frame]?'是':'否'}`;
  canvas.dataset.frame=frame;
}
function setFrame(i){frame=Math.max(0,Math.min(data.frames-1,Math.round(i)));render();}
function toggle(){if(!data)return;if(!playing&&frame===data.frames-1)frame=0;playing=!playing;last=performance.now();$('play').textContent=playing?'暂停':'播放';render();}
function animate(now){if(playing&&data){const dt=data.dt*1000/Number($('rate').value),advance=Math.floor((now-last)/dt);if(advance){last+=advance*dt;setFrame(frame+advance);if(frame===data.frames-1){playing=false;$('play').textContent='播放';}}}requestAnimationFrame(animate);}
function resize(){const r=canvas.getBoundingClientRect();width=r.width;height=r.height;const dpr=devicePixelRatio||1;canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);render();}
$('play').onclick=toggle;$('prev').onclick=()=>setFrame(frame-1);$('next').onclick=()=>setFrame(frame+1);
$('seek').oninput=e=>{if(data)setFrame(Number(e.target.value));};$('frame').oninput=e=>{if(data&&e.target.value!==''&&Number.isFinite(Number(e.target.value)))setFrame(Number(e.target.value));};
['physics','reference','joints'].forEach(id=>$(id).onchange=render);
$('view').onchange=e=>{const views={oblique:[.65,.18],front:[0,0],side:[Math.PI/2,0],top:[0,Math.PI/2]};[yaw,pitch]=views[e.target.value];render();};
$('reset').onclick=()=>{yaw=.65;pitch=.18;zoom=1;panX=panY=0;$('view').value='oblique';render();};
canvas.oncontextmenu=e=>e.preventDefault();
canvas.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,pan:e.button===2||e.shiftKey};canvas.setPointerCapture(e.pointerId);};
canvas.onpointermove=e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;if(drag.pan){panX+=dx;panY+=dy;}else{yaw+=dx*.008;pitch=Math.max(-Math.PI/2,Math.min(Math.PI/2,pitch+dy*.008));}render();};
canvas.onpointerup=()=>drag=null;canvas.onpointercancel=()=>drag=null;
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.3,Math.min(5,zoom*Math.exp(-e.deltaY*.001)));render();},{passive:false});
window.addEventListener('resize',resize);
window.addEventListener('keydown',e=>{if(['INPUT','SELECT'].includes(document.activeElement.tagName))return;if(e.code==='Space'){e.preventDefault();toggle();}if(data&&e.key==='ArrowRight')setFrame(frame+1);if(data&&e.key==='ArrowLeft')setFrame(frame-1);});
(async()=>{try{
  const response=await fetch('data.json.gz');if(!response.ok)throw Error(`HTTP ${response.status}`);
  data=await new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).json();
  mesh=unpack(data.mesh);position=unpack(data.position);reference=unpack(data.reference);faces=unpack(data.faces_u32,Uint32Array);
  center=[reference[0],reference[1]];
  if(mesh.length!==data.frames*6890*3||faces.length!==41328)throw Error('网格数据长度错误');
  $('seek').max=$('frame').max=data.frames-1;$('total').textContent=`/ ${data.frames-1}`;
  ['play','prev','next'].forEach(id=>$(id).disabled=false);
  $('status').textContent=`${data.source_key} · SMPL 6890 顶点 / 13776 三角面 · 全部 ${data.frames} 步 · 无插帧或显示平滑 · 网格/物理关节最大回算差 ${(data.native_joint_roundtrip_max_m*1000).toFixed(3)} mm`;
  resize();requestAnimationFrame(animate);
}catch(error){$('metrics').textContent='加载失败：'+error.message;$('status').textContent=String(error.stack);}})();
