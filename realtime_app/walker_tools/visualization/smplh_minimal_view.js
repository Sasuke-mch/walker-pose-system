/* Saved candidate only. Frame-indexed geometry; no fitting or display filtering. */
(async()=>{
  const response=await fetch('data.json.gz');
  if(!response.ok)throw Error(`数据请求 ${response.status}`);
  const d=await new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).json();
  const unpack=s=>new Float32Array(Uint8Array.from(atob(s),c=>c.charCodeAt(0)).buffer);
  const candidates=d.comparison?d.models:[d.models.find(model=>model.name==='repaired')||d.models[0]];
  const decoded=candidates.map(m=>({...m,decodedMesh:unpack(m.mesh),decodedPoints:unpack(m.points),decodedSole:unpack(m.sole)}));
  let m=decoded[0],mesh=m.decodedMesh,model=m.decodedPoints,sole=m.decodedSole;
  const raw=unpack(d.raw),strict=unpack(d.strict),walker=unpack(d.walker),cameras=unpack(d.cameras);
  const $=id=>document.getElementById(id),canvas=$('scene'),ctx=canvas.getContext('2d');
  const edges=[[0,1],[0,2],[1,3],[2,4],[3,5],[4,6],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
  const names=['鼻','左眼','右眼','左耳','右耳','左肩','右肩','左肘','右肘','左腕','右腕','左髋','右髋','左膝','右膝','左踝','右踝'];
  const valid=p=>p.every(Number.isFinite),point=(a,j,t=fi,n=17)=>Array.from(a.subarray((t*n+j)*3,(t*n+j)*3+3));
  const median=values=>{const v=values.filter(x=>x!==null&&Number.isFinite(x)).sort((a,b)=>a-b);return v.length?(v[Math.floor((v.length-1)/2)]+v[Math.ceil((v.length-1)/2)])/2:null;};
  const number=x=>x===null||!Number.isFinite(x)?'—':x.toFixed(1);
  const gv=p=>[p[0],-p[1],p[2]];
  const frameId=()=>d.frameIds?d.frameIds[fi]:fi;
  let fi=Math.min(61,d.frames-1),yaw=.65,pitch=.18,zoom=.9,panX=0,panY=0,playing=false,lastTime=0;
  for(const id of ['frame','seek'])$(id).max=d.frames-1;
  if(d.title){document.title=d.title;document.querySelector('h1').textContent=d.title;}
  if(d.rejection)document.querySelector('.legend').append(document.createElement('br'),document.createTextNode(d.rejection));
  const requested=new URLSearchParams(location.search).get('frame');
  if(requested!==null&&Number.isInteger(+requested)){
    const index=d.frameIds?d.frameIds.indexOf(+requested):+requested;
    if(index>=0&&index<d.frames)fi=index;
  }
  if(decoded.length>1){
    const choice=document.createElement('select');choice.id='variant';choice.setAttribute('aria-label','实验结果');
    decoded.forEach((item,i)=>choice.add(new Option(item.label,i)));
    document.querySelector('header').insertBefore(choice,$('play'));
    choice.onchange=()=>{m=decoded[+choice.value];mesh=m.decodedMesh;model=m.decodedPoints;sole=m.decodedSole;draw();};
  }
  if(d.static){$('play').disabled=true;$('play').title='独立静态解，不构成连续轨迹';$('rate').disabled=true;}
  function project(p){
    const v=gv(p),x=v[0]-d.center[0],y=v[1]+d.center[1],z=v[2]-.85;
    const xx=x*Math.cos(yaw)-y*Math.sin(yaw),yy=x*Math.sin(yaw)+y*Math.cos(yaw);
    const up=z*Math.cos(pitch)-yy*Math.sin(pitch),depth=z*Math.sin(pitch)+yy*Math.cos(pitch);
    const s=Math.min(canvas.width,canvas.height)*.85*zoom/(2+depth*.15);
    return [canvas.width*.5+panX+xx*s,canvas.height*.51+panY-up*s,depth];
  }
  function line(a,b,color,width=1,dash=[]){
    if(!valid(a)||!valid(b))return;
    const A=project(a),B=project(b);
    ctx.strokeStyle=color;ctx.lineWidth=width*devicePixelRatio;ctx.lineCap='round';ctx.setLineDash(dash.map(x=>x*devicePixelRatio));
    ctx.beginPath();ctx.moveTo(A[0],A[1]);ctx.lineTo(B[0],B[1]);ctx.stroke();ctx.setLineDash([]);
  }
  function dot(p,color,r=2){
    if(!valid(p))return;
    const q=project(p);ctx.fillStyle=color;ctx.beginPath();ctx.arc(q[0],q[1],r*devicePixelRatio,0,Math.PI*2);ctx.fill();
  }
  function drawScene(){
    const w=Math.round(canvas.clientWidth*devicePixelRatio),h=Math.round(canvas.clientHeight*devicePixelRatio);
    if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
    ctx.fillStyle='#f5f7f6';ctx.fillRect(0,0,w,h);
    const corners=[[-2.5,-2.5,0],[2.5,-2.5,0],[2.5,2.5,0],[-2.5,2.5,0]].map(project);
    ctx.fillStyle='#718279';ctx.beginPath();corners.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();ctx.fill();
    for(let u=-2.5;u<=2.5;u+=.5){line([u,-2.5,0],[u,2.5,0],'#bdcbc4',.5);line([-2.5,u,0],[2.5,u,0],'#bdcbc4',.5);}
    for(const [end,color,label] of [[[2,0,0],'#9b3030','X'],[[0,-2,0],'#18643e','Y'],[[0,0,2.2],'#34589b','Z']]){
      line([0,0,0],end,color,1.5);const p=project(end);ctx.fillStyle=color;ctx.font=`${12*devicePixelRatio}px sans-serif`;ctx.fillText(label,p[0]+5,p[1]);
    }
    for(let u=.5;u<=2;u+=.5)for(const p of [[u,0,0],[0,-u,0]]){const q=project(p);ctx.fillStyle='#283c32';ctx.font=`${10*devicePixelRatio}px sans-serif`;ctx.fillText(u.toFixed(1)+'m',q[0]+4,q[1]-4);}
    if($('trails').checked){
      const plane=d.center[0]+.8;
      for(let t=Math.max(1,fi-39);t<=fi;t++){
        for(let side=0;side<2;side++){
          const A=point(cameras,side,t-1,2),B=point(cameras,side,t,2);
          line([plane,A[1],A[2]],[plane,B[1],B[2]],'#897b96',.8);
        }
        for(const j of [15,16])if(d.accepted[t-1][j]&&d.accepted[t][j]){
          const A=point(strict,j,t-1),B=point(strict,j,t);line([A[0],A[1],.002],[B[0],B[1],.002],'#263b30',1);
        }
      }
      if($('walker').checked)for(let side=0;side<2;side++){
        const C=point(cameras,side,fi,2);line(C,[plane,C[1],C[2]],'#897b96',.7,[3,3]);
      }
    }
    if($('walker').checked){
      const n=d.walkerNames.length;
      for(const [a,b] of d.walkerEdges){const A=point(walker,a,fi,n),B=point(walker,b,fi,n);line(A,B,'#29484b',8);line(A,B,'#83a0a0',4);}
      const left=point(cameras,0,fi,2),right=point(cameras,1,fi,2);line(left,right,'#50575a',2);dot(left,'#795b87',4);dot(right,'#795b87',4);
    }
    if($('mesh').checked){
      // Project each saved vertex once, then draw the real faces from back to front.
      const p=Array.from({length:6890},(_,j)=>{const v=point(mesh,j,fi,6890);return valid(v)?project(v):null;});
      const triangles=[];
      for(const face of d.faces){const q=face.map(j=>p[j]);if(q.every(Boolean))triangles.push([q[0][2]+q[1][2]+q[2][2],q]);}
      triangles.sort((a,b)=>b[0]-a[0]);ctx.fillStyle='#84a4a0';ctx.globalAlpha=.46;
      for(const [,q] of triangles){ctx.beginPath();ctx.moveTo(q[0][0],q[0][1]);ctx.lineTo(q[1][0],q[1][1]);ctx.lineTo(q[2][0],q[2][1]);ctx.closePath();ctx.fill();}
      ctx.globalAlpha=1;
    }
    let rawEdges=0;
    if($('raw').checked){
      for(const [a,b] of edges){const A=point(raw,a),B=point(raw,b);if(valid(A)&&valid(B)){line(A,B,'#000000',1.2);rawEdges++;}}
      for(let j=0;j<17;j++){
        const p=point(raw,j);if(!valid(p))continue;
        if(d.accepted[fi][j])dot(p,'#171b19',2);
        else{const q=project(p),r=3*devicePixelRatio;ctx.strokeStyle='#b17b25';ctx.lineWidth=1.5*devicePixelRatio;ctx.beginPath();ctx.moveTo(q[0]-r,q[1]-r);ctx.lineTo(q[0]+r,q[1]+r);ctx.moveTo(q[0]+r,q[1]-r);ctx.lineTo(q[0]-r,q[1]+r);ctx.stroke();}
      }
    }
    if($('model').checked)for(let j=0;j<17;j++)dot(point(model,j),'#087970',2.4);
    if($('sole').checked)for(let j=0;j<d.soleCount;j++){const p=point(sole,j,fi,d.soleCount),ref=(m.soleReferenceMm||0)/1000;dot(p,p[2]<ref-(ref?0.005:0)?'#d31425':'#6f7526',1.3);}
    canvas.dataset.audit=JSON.stringify({frame:frameId(),sampleIndex:fi,frames:d.frames,static:!!d.static,variant:m.name,meshVariant:m.name,
      meshVertices:6890,meshFaces:d.faces.length,meshIndexEntries:d.faces.length*3,meshVisible:$('mesh').checked,
      rawEdges,modelPoints:$('model').checked?17:0,displayMap:'X,-Y,Z',interpolation:false,smoothing:false,
      ankleHeightMm:[15,16].map(j=>point(model,j)[2]*1000),soleMinMm:m.soleHeight[fi],
      hipMedianPx:median(m.errors[fi].flatMap(v=>v.slice(6,8))),stage:d.stage[fi],pose:d.pose[fi],view:{yaw,pitch,zoom,panX,panY}});
  }
  function drawInfo(){
    $('frame').min=d.frameIds?d.frameIds[0]:0;$('frame').max=d.frameIds?d.frameIds.at(-1):d.frames-1;
    $('frame').value=frameId();$('seek').value=fi;
    $('clock').textContent=`${d.time[fi].toFixed(2)} / ${d.time[d.frames-1].toFixed(2)} s`;
    const stage={stage1_walker_static_human_moving:'Stage 1 · 助步器静止',stage2_feet_static_walker_moving:'Stage 2 · 助步器移动',transition:'阶段过渡',warming_up:'准备阶段'}[d.stage[fi]]||d.stage[fi];
    const pose={held:'保持',updated:'更新',rejected:'拒绝',initialized:'初始化'}[d.pose[fi]]||d.pose[fi];
    const state=m.status[fi][0],solver={feasible_unconverged:'求解未收敛',converged:'求解收敛',not_run:'初始化',static_success:'静态求解成功退出',static_unconverged:'静态求解未收敛',static_diagnostic:'独立静态诊断'}[state]||state;
    $('status').textContent=`${m.label} · ${stage} · 地面${pose} · ${solver}${m.rejection?' · '+m.rejection:d.rejection?' · 候选未通过':''}${d.static?' · 独立静态解':''}`;
    const rejected=d.accepted[fi].flatMap((ok,j)=>ok?[]:[`${names[j]}：${d.rawReasons[fi][j]}`]);
    $('quality').textContent=`${m.label||'修正后'} · 原始帧 ${frameId()}${d.static?'；仅有 '+d.frameIds.join('/')+' 帧':''}\n${stage}\n地面变换：${d.pose[fi]}；${d.poseReason[fi]||'无拒绝原因'}\n`+
      `${solver}；${m.reason[fi][0]||'—'}\n回投中位：${number(median(m.errors[fi].flat()))} px；髋 ${number(median(m.errors[fi].flatMap(v=>v.slice(6,8))))} px\n`+
      `足底最低 L / R：${m.soleHeight[fi].map(number).join(' / ')} mm\n${m.soleReferenceMm?'鞋高目标：'+m.soleReferenceMm+' mm；允许下界：'+(m.soleReferenceMm-5)+' mm\n相对鞋高 L / R：'+m.soleClearance[fi].map(number).join(' / ')+' mm\n':''}${m.contactInfo?m.contactInfo[fi]+'\n':''}严格观测：${d.accepted[fi].filter(Boolean).length}/17\n${rejected.join('\n')}\n${m.landmarkDefinition[fi]}`;
  }
  function draw(){drawScene();drawInfo();drawImages();}
  function setFrame(value){fi=Math.max(0,Math.min(d.frames-1,Math.round(value)));playing=false;$('play').textContent='播放';draw();}
  $('frame').oninput=e=>{if(Number.isFinite(e.target.valueAsNumber)){const value=d.frameIds?d.frameIds.indexOf(e.target.valueAsNumber):e.target.valueAsNumber;if(value>=0)setFrame(value);}};
  $('frame').onchange=e=>{if(Number.isFinite(e.target.valueAsNumber)){const value=d.frameIds?d.frameIds.indexOf(e.target.valueAsNumber):e.target.valueAsNumber;if(value>=0)setFrame(value);else $('frame').value=frameId();}};
  $('seek').oninput=e=>setFrame(+e.target.value);
  $('play').onclick=()=>{if(d.static)return;playing=!playing;lastTime=0;$('play').textContent=playing?'暂停':'播放';};
  for(const id of ['mesh','raw','model','walker','trails','sole'])$(id).onchange=drawScene;
  $('view').onchange=e=>{[yaw,pitch]=({recommended:[.65,.18],side:[Math.PI/2,0],front:[0,0],top:[0,Math.PI/2]}[e.target.value]);zoom=.9;panX=panY=0;drawScene();};
  let drag=null;canvas.oncontextmenu=e=>e.preventDefault();
  canvas.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,pan:e.button===2||e.shiftKey};canvas.setPointerCapture(e.pointerId);};
  canvas.onpointerup=canvas.onpointercancel=()=>drag=null;
  canvas.onpointermove=e=>{
    if(!drag)return;
    const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;
    if(drag.pan){panX+=dx*devicePixelRatio;panY+=dy*devicePixelRatio;}
    else{yaw-=dx*.007;pitch+=dy*.007;}
    drawScene();
  };
  canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.15,Math.min(6,zoom*(e.deltaY<0?1.08:.93)));drawScene();},{passive:false});
  let imagePanel=null;
  const imageCache=new Map();
  if(d.images){
    imagePanel=document.createElement('details');imagePanel.id='projections';imagePanel.className='projections';
    imagePanel.innerHTML='<summary>同帧双目回投</summary><div><label><input id="cropLegs" type="checkbox" checked>下肢放大</label><p>黑：原身体观测　绿：模型回投　金：原趾点　紫：模型趾点</p><p id="imageFrame"></p><div class="imagePair"><canvas id="leftProjection"></canvas><canvas id="rightProjection"></canvas></div></div>';
    document.querySelector('main').append(imagePanel);imagePanel.ontoggle=drawImages;$('cropLegs').onchange=drawImages;
  }
  function drawImages(){
    if(!imagePanel||!imagePanel.open)return;
    const sample=fi,variant=m;
    $('imageFrame').textContent=`原始帧 ${frameId()} · ${m.label}${[35,47,61].includes(frameId())?' · 趾方向未参与拟合（静态组亦保持）':''}`;
    for(let side=0;side<2;side++){
      const url=d.images[fi][side];let img=imageCache.get(url);
      if(!img){img=new Image();imageCache.set(url,img);img.onload=()=>{if(sample===fi&&variant===m)drawImages();};img.onerror=()=>{$('imageFrame').textContent='原图加载失败：'+url;};img.src=url;}
      if(!img.complete||!img.naturalWidth)continue;
      const c=$(side?'rightProjection':'leftProjection'),cx=c.getContext('2d'),y0=$('cropLegs').checked?900:0;
      c.width=img.naturalWidth;c.height=img.naturalHeight-y0;cx.drawImage(img,0,y0,c.width,c.height,0,0,c.width,c.height);
      const upright=p=>side?[1079-p[1],p[0]-y0]:[p[1],1919-p[0]-y0];
      function marks(points,color,connected){
        if(!points)return;const xy=points.map(p=>p&&p.every(Number.isFinite)?upright(p):null);
        cx.strokeStyle=color;cx.fillStyle=color;cx.lineWidth=3;
        if(connected)for(const [a,b] of edges)if(xy[a]&&xy[b]){cx.beginPath();cx.moveTo(...xy[a]);cx.lineTo(...xy[b]);cx.stroke();}
        for(const p of xy)if(p){cx.beginPath();cx.arc(...p,4,0,Math.PI*2);cx.fill();}
      }
      marks(d.observedPixels[fi][side],'#000000',true);marks(m.pixels[fi][side],'#008a69',true);
      marks(d.toeObservedPixels[fi][side],'#dfae19',false);marks(m.toePixels?.[fi]?.[side],'#8c2bb1',false);
      c.dataset.frame=frameId();c.dataset.variant=m.name;
    }
  }
  function tick(time){if(playing&&time-lastTime>=1000/(30*Number($('rate').value))){fi=(fi+1)%d.frames;lastTime=time;draw();}requestAnimationFrame(tick);}
  $('loading').hidden=true;$('app').hidden=false;draw();requestAnimationFrame(tick);
  new ResizeObserver(drawScene).observe(canvas);
})().catch(error=>{document.getElementById('loading').hidden=false;document.getElementById('app').hidden=true;document.getElementById('loading').textContent='可视化加载失败：'+error.message;console.error(error);});
