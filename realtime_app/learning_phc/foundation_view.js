'use strict';
// Same-run world coordinates for every object; projection maps [X,-Y,Z].
const $=id=>document.getElementById(id),canvas=$('scene'),ctx=canvas.getContext('2d');
let bundle,data,mesh,faces,coco,rods,cameras,anchors,reference,frame=0,playing=false,last=0;
let yaw=2.5,pitch=.18,zoom=1.1,panX=0,panY=0,center,width,height,drag;
const unpack=(str,Type=Float32Array)=>{const s=atob(str),b=new Uint8Array(s.length);for(let i=0;i<s.length;i++)b[i]=s.charCodeAt(i);return new Type(b.buffer);};
function project(x,y,z){
  // Fixed projection center only: no frame-dependent translation or rescaling.
  x-=center[0]; y=-(y-center[1]); z-=center[2];
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
  const o=[0,0,0],p=project(...o);
  [[.5,0,0,'X / 0.5m'],[0,.5,0,'Y / 0.5m'],[0,0,.5,'Z / 0.5m']].forEach(a=>{const q=project(o[0]+a[0],o[1]+a[1],a[2]);line(p,q,'#46554d',2);ctx.fillStyle='#46554d';ctx.font='12px system-ui';ctx.fillText(a[3],q[0]+5,q[1]-5);});
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

function at(array,index){return project(...array.subarray(index,index+3));}
function marker(point,color,radius=3){ctx.fillStyle=color;ctx.beginPath();ctx.arc(point[0],point[1],radius,0,Math.PI*2);ctx.fill();}
function members(){
  const scale=Math.min(width/3.4,height/2.5)*zoom;
  for(let j=0;j<bundle.radii.length;j++){
    const k=frame*bundle.radii.length*6+j*6,a=at(rods,k),b=at(rods,k+3),r=Math.max(1,bundle.radii[j]*scale);
    ctx.lineCap='round';line(a,b,'#414b44',2*r+1.2);line(a,b,'#88948b',Math.max(1,2*r-1));
    marker(a,'#778179',r);marker(b,'#778179',r);
  }
  ctx.lineCap='butt';
}
function details(){
  const c=frame*6,left=at(cameras,c),right=at(cameras,c+3);
  line(left,right,'#70418b',2);marker(left,'#70418b',4);marker(right,'#70418b',4);
  ctx.font='12px system-ui';ctx.fillStyle='#70418b';ctx.fillText('左相机',left[0]+5,left[1]+12);ctx.fillText('右相机',right[0]+5,right[1]-7);
  if($('grips').checked)for(let j=0;j<2;j++){
    const k=frame*12+j*6,a=at(anchors,k),b=at(anchors,k+3);
    line(a,b,'#b02a26',2.4);marker(a,'#b02a26',3);marker(b,'#b02a26',3);
  }
  if($('coco').checked)for(let j=0;j<17;j++)marker(at(coco,frame*51+j*3),'#dc892b',2.6);
  if(reference&&$('target').checked){
    // Author 24-body topology; reference model, never an observed skeleton.
    const parents=[-1,0,1,2,3,0,5,6,7,0,9,10,11,12,11,14,15,16,17,11,19,20,21,22];
    for(let j=1;j<24;j++)line(at(reference,frame*72+j*3),at(reference,frame*72+parents[j]*3),'#c77c25',1.7);
  }
}
function render(){
 if(!data)return;
 ctx.clearRect(0,0,width,height);ctx.fillStyle='#f6f6f2';ctx.fillRect(0,0,width,height);
 grid();members();if($('physics').checked)surface();details();
 $('seek').value=$('frame').value=frame;
 const gap=data.gap_mm[frame],force=data.force_N[frame];
 const f=force.map(v=>v===null?'该末态无后续施力':v.toFixed(1)+' N').join(' / ');
 $('metrics').textContent=`${data.label} · 状态 ${frame} / ${data.frames-1} · t=${data.time[frame].toFixed(6)} s　握点间隙 左/右 ${gap[0].toFixed(2)} / ${gap[1].toFixed(2)} mm　${data.force_label||'下一步施加力'} 左/右 ${f}`;
 if(data.fallen)$('metrics').textContent+=`　${data.fallen[frame]?'摔倒代理触发（失败保留）':'未触发摔倒代理'}；有限回投中位差 ${data.errors_median_px[frame]===null?'不可用':data.errors_median_px[frame].toFixed(1)+' px'}（非3D真值误差）`;
 if(data.target_index)$('metrics').textContent+=`　已到达目标帧 ${data.target_index[frame]}，年龄 ${data.target_age_ms[frame].toFixed(1)} ms；根位置与参考差 ${data.root_reference_difference_mm[frame].toFixed(1)} mm（非真值误差）`;
 $('status').textContent=`${data.description} · ${data.frames}个保存状态 · 控制240Hz / 物理480Hz · 6890顶点 / 13776真实三角面 · 无图像/Stage观测；橙点仅为模型点 · 首态原人体提高9.69mm，物理地面不动`;
 if(data.status_text)$('status').textContent=data.status_text+' · 6890顶点 / 13776真实三角面 · 地面Z=0';
 canvas.dataset.frame=frame;canvas.dataset.run=data.key;
 if(reference)$('status').textContent+=' · 橙骨架为模型参考；能运动不等于跟踪通过';
}
function setFrame(i){frame=Math.max(0,Math.min(data.frames-1,Math.round(i)));render();}
function choose(key){
 playing=false;$('play').textContent='播放';const oldTime=data?data.time[frame]:0;
 data=bundle.runs.find(r=>r.key===key);mesh=unpack(data.mesh);rods=unpack(data.rods);cameras=unpack(data.cameras);anchors=unpack(data.anchors);coco=unpack(data.coco);
 reference=data.reference?unpack(data.reference):null;$('target-label').hidden=!reference;
 if(mesh.length!==data.frames*6890*3||rods.length!==data.frames*78||cameras.length!==data.frames*6)throw Error('状态/网格/场景长度不一致');
 frame=data.time.reduce((best,t,i)=>Math.abs(t-oldTime)<Math.abs(data.time[best]-oldTime)?i:best,0);
 $('seek').max=$('frame').max=data.frames-1;$('total').textContent=`/ ${data.frames-1}`;render();
}
function toggle(){if(!data)return;if(!playing&&frame===data.frames-1)frame=0;playing=!playing;last=performance.now();$('play').textContent=playing?'暂停':'播放';render();}
function animate(now){if(playing&&data){const elapsed=(now-last)*Number($('rate').value)/1000;let next=frame;while(next+1<data.frames&&data.time[next+1]-data.time[frame]<=elapsed)next++;if(next>frame){last+=(data.time[next]-data.time[frame])*1000/Number($('rate').value);setFrame(next);if(frame===data.frames-1){playing=false;$('play').textContent='播放';}}}requestAnimationFrame(animate);}
function resize(){const r=canvas.getBoundingClientRect();width=r.width;height=r.height;const dpr=devicePixelRatio||1;canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);render();}
$('play').onclick=toggle;$('prev').onclick=()=>setFrame(frame-1);$('next').onclick=()=>setFrame(frame+1);
$('seek').oninput=e=>setFrame(Number(e.target.value));$('frame').oninput=e=>{if(e.target.value!==''&&Number.isFinite(Number(e.target.value)))setFrame(Number(e.target.value));};
$('run').onchange=e=>choose(e.target.value);['physics','coco','grips','target'].forEach(id=>$(id).onchange=render);
$('view').onchange=e=>{[yaw,pitch]={oblique:[2.5,.18],front:[0,0],side:[Math.PI/2,0],top:[0,Math.PI/2]}[e.target.value];render();};
$('reset').onclick=()=>{yaw=2.5;pitch=.18;zoom=1.1;panX=panY=0;$('view').value='oblique';render();};
canvas.oncontextmenu=e=>e.preventDefault();canvas.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,pan:e.button===2||e.shiftKey};canvas.setPointerCapture(e.pointerId);};
canvas.onpointermove=e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;if(drag.pan){panX+=dx;panY+=dy;}else{yaw+=dx*.008;pitch+=dy*.008;}render();};
canvas.onpointerup=canvas.onpointercancel=()=>drag=null;
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.2,Math.min(7,zoom*Math.exp(-e.deltaY*.001)));render();},{passive:false});
window.addEventListener('resize',resize);window.addEventListener('keydown',e=>{if(['INPUT','SELECT'].includes(document.activeElement.tagName))return;if(e.code==='Space'){e.preventDefault();toggle();}if(data&&e.key==='ArrowRight')setFrame(frame+1);if(data&&e.key==='ArrowLeft')setFrame(frame-1);});
(async()=>{try{
 const response=await fetch('data.json.gz');if(!response.ok)throw Error('HTTP '+response.status);
 bundle=await new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).json();faces=unpack(bundle.faces_u32,Uint32Array);center=bundle.center;
 if(faces.length!==41328)throw Error('真实三角面索引长度错误');
 for(const run of bundle.runs){const option=document.createElement('option');option.value=run.key;option.textContent=run.label;$('run').appendChild(option);}
 ['run','play','prev','next'].forEach(id=>$(id).disabled=false);$('rate').value=String(bundle.default_rate||.1);choose(bundle.runs[0].key);resize();requestAnimationFrame(animate);
}catch(error){$('metrics').textContent='加载失败：'+error.message;$('metrics').className='error';$('status').textContent=String(error.stack);console.error(error);}})();
