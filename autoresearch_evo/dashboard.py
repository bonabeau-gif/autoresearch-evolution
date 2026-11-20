"""Generate one dependency-free offline HTML experiment dashboard."""
from __future__ import annotations

import json
from pathlib import Path

from .core import read_population


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Autoresearch Evolution | Experiment Dashboard</title>
<style>
:root{color-scheme:dark;--bg:#0c111b;--panel:#141d2c;--line:#2a3a4f;--text:#e6edf6;--muted:#9aaec7;--a:#65d8be;--b:#79a9fa;--bad:#ef8d88}
*{box-sizing:border-box}body{font:14px/1.6 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto;margin:0;background:var(--bg);color:var(--text)}
main{max-width:1220px;margin:auto;padding:32px 24px 60px}.eyebrow{font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:var(--a);font-size:11px}
h1{font-size:clamp(28px,3.8vw,45px);margin:4px 0 6px;letter-spacing:-.04em}p{color:var(--muted);margin-top:0}.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:28px 0}
.metric,.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px}.metric{padding:20px}.value{font-size:25px;font-variant-numeric:tabular-nums;font-weight:700;letter-spacing:-.03em}.name{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em}
.grid{display:grid;grid-template-columns:1.25fr 1fr;gap:16px}.panel{padding:22px;margin-bottom:16px;overflow:hidden}h2{font-size:16px;margin:0 0 8px}svg{display:block;width:100%;height:auto;overflow:visible}.legend{display:flex;gap:22px;font-size:12px;color:var(--muted)}.key{height:9px;width:9px;border-radius:50%;display:inline-block;margin-right:5px}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:8px 0 12px}select,input{background:#0c1523;border:1px solid var(--line);color:var(--text);border-radius:8px;padding:9px 12px}input{min-width:200px;flex:1}
.tablewrap{overflow:auto;max-height:460px}table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}th{text-align:left;position:sticky;top:0;background:#202e43;color:#bbccdd;padding:10px 12px;cursor:pointer;white-space:nowrap}td{border-bottom:1px solid var(--line);padding:8px 12px;white-space:nowrap}td:nth-child(8){min-width:240px;white-space:normal}.status-ok{color:var(--a)}.status-crash,.status-timeout{color:var(--bad)}
.node{cursor:pointer;transition:opacity .1s}.edge{opacity:.28;stroke:#7d94ad}.node:hover{stroke:#fff;stroke-width:2.5}.detail{color:var(--muted);background:#0c1523;border-radius:8px;border:1px solid var(--line);padding:10px 12px;min-height:56px;font-size:12px}
footer{color:var(--muted);font-size:12px;padding-top:18px}@media(max-width:800px){.grid{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}}
</style></head><body><main>
<div class="eyebrow">Autonomous research • Evolutionary search</div><h1>Population, not just a champion.</h1>
<p>Validation bits per byte is the sole fitness objective. Every node below is a committed <code>train.py</code> genome; carried elites are never reevaluated.</p>
<div id="metrics" class="stats"></div>
<div class="grid"><section class="panel"><h2>Validation quality over generations</h2><p>Lower is better. Elites are included in generation medians.</p><svg id="chart" viewBox="0 0 670 350" aria-label="Validation performance trend"></svg>
<div class="legend"><span><i class="key" style="background:var(--a)"></i>Best so far</span><span><i class="key" style="background:var(--b)"></i>Generation median</span></div></section>
<section class="panel"><h2>Evolutionary lineage</h2><p>Columns are generations; links connect mutants and crossover descendants to their parents. Click a node.</p><svg id="lineage" viewBox="0 0 530 350" aria-label="Evolutionary population graph"></svg><div class="detail" id="detail">Select a node to inspect parentage and result.</div></section></div>
<section class="panel"><h2>Individuals and experiments</h2><div class="controls"><select id="generation"><option value="">All generations</option></select><select id="status"><option value="">All statuses</option><option>ok</option><option>crash</option><option>timeout</option></select><input id="search" placeholder="Search commits, mutations and parents"></div>
<div class="tablewrap"><table id="table"><thead><tr><th data-sort="gen">Gen</th><th data-sort="slot">Slot</th><th data-sort="commit">Commit</th><th data-sort="status">Status</th><th data-sort="val_bpb">val_bpb ↓</th><th data-sort="memory_gb">VRAM GiB</th><th data-sort="method">Method</th><th data-sort="description">Description</th><th data-sort="parent1">Parent 1</th><th data-sort="parent2">Parent 2</th></tr></thead><tbody></tbody></table></div></section>
<footer>Generated from untracked <code>population.tsv</code> and <code>results.tsv</code>. Offline and self-contained. Graph values are measured logs; no synthetic results are supplied.</footer>
</main><script>
const rows=__DATA__;
const $=s=>document.querySelector(s);const svg=(tag,attrs={})=>{let e=document.createElementNS('http://www.w3.org/2000/svg',tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,String(v)));return e};
const valid=rows.filter(r=>r.status==='ok'&&r.val_bpb!==null),gens=[...new Set(rows.map(x=>x.gen))].sort((a,b)=>a-b);const vals=valid.map(r=>r.val_bpb);const minval=vals.length?Math.min(...vals):null;
$('#metrics').innerHTML=[['Global best',minval?.toFixed(6)||'—'],['Generations',gens.length],['Evaluations',rows.filter(r=>r.method!=='elite').length],['Successful',rows.filter(r=>r.status==='ok'&&r.method!=='elite').length]].map(([k,v])=>`<div class="metric"><div class="name">${k}</div><div class="value">${v}</div></div>`).join('');
function lineChart(){let g=$('#chart');g.replaceChildren();if(!valid.length)return;let W=670,H=350,px=50,py=30;let ymin=Math.min(...vals),ymax=Math.max(...vals);let pad=Math.max(.00001,(ymax-ymin)*.13);ymin-=pad;ymax+=pad;let x=d=>px+(d-(gens[0]||0))/Math.max(1,gens.at(-1)-gens[0])*(W-75),y=v=>H-45-(v-ymin)/(ymax-ymin)*(H-85);
for(let i=0;i<5;i++){let yy=py+i*(H-py-45)/4,mark=ymax-(yy-py)/(H-py-45)*(ymax-ymin);g.append(svg('line',{x1:px,y1:yy,x2:W-20,y2:yy,stroke:'#26364d'}));let t=svg('text',{x:2,y:yy+4,fill:'#97abc4','font-size':11});t.textContent=mark.toFixed(4);g.append(t)}
gens.forEach(d=>{let t=svg('text',{x:x(d),y:H-15,fill:'#97abc4','text-anchor':'middle','font-size':11});t.textContent=d;g.append(t)});
let best=Infinity;let trend=gens.map(gen=>{let points=valid.filter(r=>r.gen===gen).map(r=>r.val_bpb).sort((a,b)=>a-b);if(points.length)best=Math.min(best,points[0]);return {gen,best,median:points.length?points[Math.floor(points.length/2)]:null}});
for(let [field,color] of [['best','#65d8be'],['median','#79a9fa']]){let points=trend.filter(d=>d[field]!==null&&Number.isFinite(d[field]));if(!points.length)continue;g.append(svg('polyline',{points:points.map(d=>`${x(d.gen)},${y(d[field])}`).join(' '),stroke:color,'stroke-width':3,fill:'none','stroke-linejoin':'round'}));points.forEach(d=>{let c=svg('circle',{cx:x(d.gen),cy:y(d[field]),r:4,fill:color});let title=svg('title');title.textContent=`gen ${d.gen}: ${field} ${d[field].toFixed(6)}`;c.append(title);g.append(c)})}}
function lineage(){let g=$('#lineage');g.replaceChildren();if(!rows.length)return;let maxSlots=Math.max(...rows.map(r=>r.slot))+1,stepX=gens.length>1?440/(gens.length-1):0,stepY=280/Math.max(1,maxSlots-1),pos=new Map();rows.forEach(r=>pos.set(r.gen+':'+r.slot,{x:45+(r.gen-gens[0])*stepX,y:35+r.slot*stepY}));
// Only connect ancestors from earlier generations. Generation-zero baseline is a shared root.
for(let r of rows){if(!r.gen)continue;let end=pos.get(r.gen+':'+r.slot);for(let parent of [r.parent1,r.parent2]){if(!parent)continue;let candidates=rows.filter(z=>z.commit===parent&&z.gen<r.gen),prev=candidates.sort((a,b)=>b.gen-a.gen)[0];if(!prev)continue;let begin=pos.get(prev.gen+':'+prev.slot);g.append(svg('line',{x1:begin.x,y1:begin.y,x2:end.x,y2:end.y,class:'edge','stroke-width':r.method==='crossover'?2:1.5}))}}
rows.forEach(r=>{let p=pos.get(r.gen+':'+r.slot);let node=svg('circle',{cx:p.x,cy:p.y,r:r.method==='elite'?5:7,fill:r.status==='ok'?'#65d8be':'#ef8d88',class:'node',opacity:r.method==='elite'?.7:1});node.addEventListener('click',()=>{$('#detail').textContent=`Generation ${r.gen} slot ${r.slot} | ${r.commit} | ${r.status} | val_bpb ${r.val_bpb??'n/a'} | ${r.method} | ${r.description} | parents ${r.parent1||'—'} ${r.parent2||''}`});let title=svg('title');title.textContent=r.commit+' | '+r.description;node.append(title);g.append(node)});
}
let sortKey='gen',sortDir=-1;for(let gen of gens){let opt=document.createElement('option');opt.value=gen;opt.textContent='Generation '+gen;$('#generation').append(opt)}
function renderTable(){let gen=$('#generation').value,status=$('#status').value,q=$('#search').value.toLowerCase();let data=rows.filter(r=>(!gen||String(r.gen)===gen)&&(!status||r.status===status)&&(!q||Object.values(r).join(' ').toLowerCase().includes(q)));
data.sort((a,b)=>{let av=a[sortKey]??'',bv=b[sortKey]??'';return sortDir*(typeof av==='number'&&typeof bv==='number'?av-bv:String(av).localeCompare(String(bv)))});
let tbody=$('#table tbody');tbody.replaceChildren();for(let r of data){let tr=document.createElement('tr');for(let k of ['gen','slot','commit','status','val_bpb','memory_gb','method','description','parent1','parent2']){let td=document.createElement('td');td.textContent=r[k]===null?'':(k==='val_bpb'?Number(r[k]).toFixed(6):r[k]);if(k==='status')td.className='status-'+r[k];tr.append(td)}tbody.append(tr)}}
$('#generation').onchange=$('#status').onchange=$('#search').oninput=renderTable;document.querySelectorAll('th[data-sort]').forEach(t=>t.onclick=()=>{if(sortKey===t.dataset.sort)sortDir*=-1;else{sortKey=t.dataset.sort;sortDir=1}renderTable()});lineChart();lineage();renderTable();
</script></body></html>'''


def build_dashboard(root: Path) -> Path:
    rows = read_population(root / "population.tsv")
    data = [{"gen": r.gen, "slot": r.slot, "commit": r.commit, "status": r.status,
             "val_bpb": r.val_bpb, "memory_gb": r.memory_gb,
             "parent1": r.parent1, "parent2": r.parent2,
             "method": r.method, "description": r.description} for r in rows]
    # Prevent user-generated descriptions from closing the embedded JSON script.
    serial = json.dumps(data, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")
    target = root / "reports" / "progress.html"
    target.parent.mkdir(exist_ok=True)
    target.write_text(HTML.replace("__DATA__", serial), encoding="utf-8")
    return target
