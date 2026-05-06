import os, json, joblib
import numpy as np
from flask import Flask, request, render_template_string

BASE = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# Load model files
model       = joblib.load(os.path.join(BASE, "model.pkl"))
model_org   = joblib.load(os.path.join(BASE, "model_organism.pkl"))
scaler      = joblib.load(os.path.join(BASE, "scaler.pkl"))
le_genus    = joblib.load(os.path.join(BASE, "le_genus.pkl"))
le_org      = joblib.load(os.path.join(BASE, "le_org.pkl"))
test_names  = joblib.load(os.path.join(BASE, "test_names.pkl"))

with open(os.path.join(BASE, "organism_profiles.json"), encoding="utf-8") as f:
    PROFILES = json.load(f)
with open(os.path.join(BASE, "summary.json")) as f:
    SUMMARY = json.load(f)

SYM2NUM = {"+":4,"(+)":3,"d":2,"(-)":1,"-":0,"v":2,"ND":2,"":2}
SYM_CLR = {"+":"#27ae60","(+)":"#2ecc71","d":"#f39c12","(-)":"#e67e22",
           "-":"#e74c3c","ND":"#95a5a6","v":"#9b59b6"}

GENUS_DESCS = {
    "Chitrobacter":  "Gram-negative, facultatively anaerobic rods. Found in soil, water, and intestinal tract.",
    "Enterobacter":  "Gram-negative, motile rods. Important nosocomial pathogen with antibiotic resistance.",
    "Erwinia":       "Gram-negative rods primarily causing plant diseases.",
    "Escherichia":   "Gram-negative rods. Most strains are gut commensals; pathogenic strains cause UTIs and septicemia.",
    "Klebsiella":    "Gram-negative, encapsulated rods. Major cause of hospital-acquired infections.",
    "Morganella":    "Gram-negative rods. Associated with UTIs and nosocomial bacteremia.",
    "proteus":       "Gram-negative, highly motile rods. Common cause of catheter-associated UTIs.",
    "Providencia":   "Gram-negative rods. Associated with UTIs in catheterized patients.",
    "Salmonella":    "Gram-negative rods. Major foodborne pathogens causing salmonellosis and typhoid.",
    "Serratia":      "Gram-negative rods. Important ICU pathogen with high antibiotic resistance.",
    "shigella":      "Gram-negative, non-motile rods. Cause shigellosis (bacillary dysentery).",
    "Yersinia ":     "Gram-negative rods including plague bacillus and gastroenteritis-causing species.",
}

KEY_TESTS = ["Indole","Methyl Red","voges-proskauer","citrate","urease",
             "h2s production","oxidase","Motility","lysine decarboxylase",
             "ornithine decarboxylase","Gram staining","sucrose","lactose",
             "ONPG","phenylalanine deaminase","esculin hydrolysis"]

ALL_TEST_SUGGESTIONS = list(set(test_names + [
    "Catalase","Coagulase","Hemolysis","Spore formation","Nitrate reduction",
    "Beta-galactosidase","DNase","Lipase","Amylase","Gelatin hydrolysis",
]))

def ml_identify(user_tests_dict):
    vals  = [SYM2NUM.get(str(user_tests_dict.get(t,"d")).strip(), 2) for t in test_names]
    X     = np.array(vals).reshape(1, -1)
    Xsc   = scaler.transform(X)
    g_idx = model.predict(Xsc)[0]
    g_name  = le_genus.inverse_transform([g_idx])[0]
    g_proba = model.predict_proba(Xsc)[0]
    g_conf  = round(float(g_proba.max()) * 100, 1)
    o_idx   = model_org.predict(Xsc)[0]
    o_name  = le_org.inverse_transform([o_idx])[0]
    o_proba = model_org.predict_proba(Xsc)[0]
    top5    = np.argsort(o_proba)[::-1][:5]
    cands   = [{"organism": le_org.inverse_transform([i])[0],
                "confidence": round(float(o_proba[i]) * 100, 1),
                "genus": PROFILES.get(le_org.inverse_transform([i])[0], {}).get("genus", "?")}
               for i in top5]
    profile   = PROFILES.get(o_name, {}).get("profile", {})
    entered   = {t: v for t, v in user_tests_dict.items() if t in test_names}
    matches   = sum(1 for t, v in entered.items()
                    if abs(SYM2NUM.get(str(profile.get(t,"d")).strip(), 2) -
                           SYM2NUM.get(str(v).strip(), 2)) <= 1)
    match_pct = round(matches / len(entered) * 100, 1) if entered else 0
    return {
        "organism":   o_name,
        "genus":      g_name,
        "genus_conf": g_conf,
        "org_conf":   round(float(o_proba[o_idx]) * 100, 1),
        "candidates": cands,
        "match_pct":  match_pct,
        "genus_desc": GENUS_DESCS.get(g_name, "Member of Enterobacteriaceae."),
        "genus_orgs": [o for o, d in PROFILES.items() if d.get("genus") == g_name],
        "profile":    profile,
    }

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Enterobacteriaceae Identifier</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#f0f4f8;color:#2c3e50}
.hdr{background:linear-gradient(135deg,#1a252f,#1565c0);color:#fff;padding:16px 28px;display:flex;align-items:center;gap:12px}
.hdr h1{font-size:18px;font-weight:700}.hdr p{font-size:11px;color:#90caf9;margin-top:2px}
.stats{background:#1565c0;padding:7px 28px;display:flex;gap:24px;font-size:12px;color:#bbdefb;flex-wrap:wrap}
.stats b{color:#fff}
.main{max-width:1350px;margin:0 auto;padding:18px 14px;display:grid;grid-template-columns:360px 1fr;gap:16px}
.card{background:#fff;border-radius:12px;padding:16px;box-shadow:0 2px 10px rgba(0,0,0,.07);margin-bottom:12px}
.ctitle{font-size:11px;font-weight:700;color:#78909c;text-transform:uppercase;letter-spacing:.7px;margin-bottom:10px;padding-bottom:7px;border-bottom:2px solid #f0f4f8}
.quick{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:10px}
.qtag{font-size:11px;padding:4px 10px;border-radius:20px;border:1.5px solid #e0e6ef;cursor:pointer;background:#fafbfc;color:#546e7a}
.qtag:hover{background:#e8f0fe;border-color:#1565c0;color:#1565c0}
.tgrid{max-height:380px;overflow-y:auto;display:flex;flex-direction:column;gap:5px;padding-right:2px}
.tgrid::-webkit-scrollbar{width:4px}
.tgrid::-webkit-scrollbar-thumb{background:#cfd8dc;border-radius:4px}
.trow{display:flex;align-items:center;gap:7px;padding:6px 9px;border-radius:9px;border:1.5px solid #f0f4f8;background:#fafbfc}
.trow:hover{border-color:#1565c0}
.trow.active{border-color:#27ae60;background:#e8f5e9}
.tname{flex:1;font-size:12px;font-weight:500;color:#37474f}
.tsel{padding:4px 5px;border:1px solid #e0e6ef;border-radius:6px;font-size:11px;background:#fff;cursor:pointer;min-width:110px}
.tsel:focus{outline:none;border-color:#1565c0}
.delbtn{background:none;border:none;cursor:pointer;color:#b0bec5;font-size:14px;padding:2px 5px;border-radius:4px}
.delbtn:hover{color:#e53935;background:#ffebee}
.addrow{display:flex;gap:6px;margin-top:8px}
.addrow input{flex:1;padding:7px 10px;border:1.5px solid #e0e6ef;border-radius:8px;font-size:12px}
.addrow input:focus{outline:none;border-color:#1565c0}
.addrow select{padding:7px 5px;border:1.5px solid #e0e6ef;border-radius:8px;font-size:11px;min-width:108px}
.addbtn{padding:7px 14px;background:#1565c0;color:#fff;border:none;border-radius:8px;font-size:12px;font-weight:700;cursor:pointer}
.runbtn{width:100%;padding:12px;border:none;border-radius:10px;font-size:14px;font-weight:700;cursor:pointer;margin-top:8px;background:#1565c0;color:#fff}
.runbtn:hover{opacity:.88}
.runbtn:disabled{opacity:.5;cursor:not-allowed}
.clrbtn{width:100%;padding:7px;background:#fff;color:#90a4ae;border:1.5px solid #e0e6ef;border-radius:10px;font-size:12px;cursor:pointer;margin-top:5px}
.hero{background:linear-gradient(135deg,#1a252f,#1565c0);border-radius:13px;padding:20px;color:#fff;margin-bottom:14px}
.hero-org{font-size:22px;font-weight:700;font-style:italic;margin-bottom:3px}
.hero-sub{font-size:13px;color:#90caf9;margin-bottom:14px}
.hstats{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.hstat{background:rgba(255,255,255,.12);border-radius:9px;padding:10px;text-align:center}
.hstat-v{font-size:18px;font-weight:700}
.hstat-l{font-size:10px;color:rgba(255,255,255,.65);margin-top:2px}
.cbar{height:6px;background:rgba(255,255,255,.15);border-radius:20px;overflow:hidden;margin-top:10px}
.cfill{height:100%;border-radius:20px;background:linear-gradient(90deg,#4caf50,#81c784)}
.tabs{display:flex;gap:4px;margin-bottom:12px;flex-wrap:wrap}
.tab{padding:6px 14px;border-radius:8px;border:1.5px solid #e0e6ef;font-size:11px;font-weight:700;cursor:pointer;background:#fff;color:#78909c}
.tab.on{background:#1565c0;color:#fff;border-color:#1565c0}
.tp{display:none}.tp.on{display:block}
.cands{display:flex;flex-direction:column;gap:6px}
.cand{display:flex;align-items:center;gap:9px;padding:9px 12px;border-radius:9px;border:1.5px solid #f0f4f8}
.cand.top{border-color:#27ae60;background:#f0fff4}
.crank{width:25px;height:25px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:#fff;flex-shrink:0}
.cname{font-size:12px;font-weight:600;font-style:italic}
.csub{font-size:11px;color:#78909c;margin-top:1px}
.cpct{font-size:13px;font-weight:700;flex-shrink:0}
.cbar2{flex:1;height:4px;background:#f0f4f8;border-radius:4px;overflow:hidden}
.cfill2{height:100%;border-radius:4px;background:#1565c0}
.ptbl{width:100%;border-collapse:collapse;font-size:12px}
.ptbl th{background:#f8fafc;padding:7px 9px;text-align:left;font-weight:700;color:#78909c;font-size:10px;text-transform:uppercase;border-bottom:2px solid #f0f4f8}
.ptbl td{padding:6px 9px;border-bottom:1px solid #f8fafc}
.ptbl tr:hover td{background:#f8fafc}
.vbadge{display:inline-block;padding:2px 7px;border-radius:12px;font-size:11px;font-weight:600}
.tag-wrap{display:flex;flex-wrap:wrap;gap:5px}
.tag{font-size:11px;padding:3px 9px;border-radius:12px}
.igrid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px}
.ibox{background:#f8fafc;border-radius:8px;padding:9px 11px;border:1.5px solid #f0f4f8}
.il{font-size:10px;color:#90a4ae;text-transform:uppercase;letter-spacing:.5px}
.iv{font-size:12px;font-weight:600;margin-top:2px}
.org-chips{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px}
.ochip{background:#f0f4f8;padding:4px 10px;border-radius:20px;font-size:11px;font-style:italic}
.ochip.me{background:#c8e6c9;color:#2e7d32;font-weight:700}
.empty{text-align:center;padding:50px 20px;color:#b0bec5}
.scroll{max-height:360px;overflow-y:auto}
.srch{margin-bottom:8px}
.srch input{width:100%;padding:7px 10px;border:1.5px solid #e0e6ef;border-radius:8px;font-size:12px}
.hint{font-size:11px;color:#90a4ae;margin-top:7px;text-align:center}
@media(max-width:860px){.main{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="hdr">
  <div style="font-size:26px">&#127850;</div>
  <div>
    <h1>Enterobacteriaceae Identification System</h1>
    <p>ML-powered bacterial identification &bull; Dissertation Research Tool</p>
  </div>
</div>
<div class="stats">
  <div><b>{{summary.n_organisms}}</b> organisms</div>
  <div><b>{{summary.n_genera}}</b> genera</div>
  <div><b>{{summary.n_tests}}</b> tests</div>
  <div>Model: <b>{{summary.best_model}}</b></div>
  <div>Accuracy: <b>{{summary.loo_accuracy}}%</b> LOO-CV</div>
</div>
<div class="main">
<div>
  <div class="card">
    <div class="ctitle">Quick add key tests</div>
    <div class="quick">
      {% for t in key_tests %}<div class="qtag" onclick="quickAdd('{{t}}')">{{t}}</div>{% endfor %}
    </div>
  </div>
  <div class="card">
    <div class="ctitle">Biochemical test results <span id="cnt" style="background:#e8f0fe;color:#1565c0;border-radius:20px;padding:2px 8px;font-size:11px;font-weight:700">0</span></div>
    <form method="POST" action="/identify" id="frm">
      <div class="tgrid" id="tgrid">
        {% for t, v in entered.items() %}
        <div class="trow active" id="row-{{loop.index}}">
          <span class="tname">{{t}}</span>
          <select name="test_{{t}}" class="tsel" onchange="markA(this,'row-{{loop.index}}')">
            {% for sym in ['+','-','(+)','(-)','d','ND'] %}<option value="{{sym}}" {{'selected' if v==sym else ''}}>{{sym}}</option>{% endfor %}
          </select>
          <button type="button" class="delbtn" onclick="delRow(this)">&#10005;</button>
          <input type="hidden" name="testname_{{loop.index}}" value="{{t}}">
        </div>
        {% endfor %}
      </div>
    </form>
    <div class="addrow">
      <input type="text" id="newtname" placeholder="Type any test name..." list="tsugg" autocomplete="off">
      <datalist id="tsugg">{% for t in all_tests %}<option value="{{t}}">{% endfor %}</datalist>
      <select id="newtsym">
        <option value="+">+ Positive</option>
        <option value="-">- Negative</option>
        <option value="(+)">(+) Weak+</option>
        <option value="(-)"  >(-) Weak-</option>
        <option value="d" selected>d Variable</option>
        <option value="ND">ND No data</option>
      </select>
      <button class="addbtn" onclick="addTest()">+ Add</button>
    </div>
    <button type="button" id="runbtn" class="runbtn" onclick="document.getElementById('frm').submit()" disabled>&#128300; Identify Organism</button>
    <button type="button" class="clrbtn" onclick="clearAll()">Clear all</button>
    <div class="hint">Enter at least 3 tests for best accuracy</div>
  </div>
</div>
<div>
{% if result %}
<div class="hero">
  <div style="font-size:10px;color:rgba(255,255,255,.7);text-transform:uppercase;letter-spacing:.8px;margin-bottom:4px">Predicted organism</div>
  <div class="hero-org">{{result.organism}}</div>
  <div class="hero-sub">Genus: <b>{{result.genus}}</b> &bull; Profile match: {{result.match_pct}}%</div>
  <div class="hstats">
    <div class="hstat"><div class="hstat-v">{{result.genus_conf}}%</div><div class="hstat-l">Genus conf.</div></div>
    <div class="hstat"><div class="hstat-v">{{result.org_conf}}%</div><div class="hstat-l">Organism conf.</div></div>
    <div class="hstat"><div class="hstat-v">{{n_entered}}</div><div class="hstat-l">Tests entered</div></div>
    <div class="hstat"><div class="hstat-v">{{result.match_pct}}%</div><div class="hstat-l">Profile match</div></div>
  </div>
  <div class="cbar"><div class="cfill" style="width:{{result.genus_conf}}%"></div></div>
</div>
<div class="card">
  <div class="tabs">
    <button class="tab on" onclick="sw(this,'t1')">&#127942; Top Candidates</button>
    <button class="tab" onclick="sw(this,'t2')">&#128203; Test Profile</button>
    <button class="tab" onclick="sw(this,'t3')">&#128196; Genus Info</button>
    <button class="tab" onclick="sw(this,'t4')">&#127775; Attributes</button>
  </div>
  <div id="t1" class="tp on">
    <div class="cands">
      {% for c in result.candidates %}
      <div class="cand {{'top' if loop.first else ''}}">
        <div class="crank" style="background:{{['#27ae60','#1565c0','#9c27b0','#e67e22','#78909c'][loop.index0]}}">{{loop.index}}</div>
        <div style="flex:1"><div class="cname">{{c.organism}}</div><div class="csub">{{c.genus}}</div></div>
        <div class="cbar2"><div class="cfill2" style="width:{{[100,70,50,30,15][loop.index0]}}%"></div></div>
        <div class="cpct">{{c.confidence}}%</div>
      </div>
      {% endfor %}
    </div>
    <p style="font-size:11px;color:#90a4ae;margin-top:10px">Model: {{summary.best_model}} &bull; LOO-CV: {{summary.loo_accuracy}}%</p>
  </div>
  <div id="t2" class="tp">
    <div class="srch"><input type="text" id="psrch" placeholder="Search test..." oninput="filterTbl()"></div>
    <div class="scroll">
      <table class="ptbl" id="ptbl">
        <thead><tr><th>Test</th><th>Known result</th><th>Your input</th><th>Match</th></tr></thead>
        <tbody>
          {% for row in profile_rows %}
          <tr>
            <td>{{row.test}}</td>
            <td><span class="vbadge" style="background:{{row.kclr}}22;color:{{row.kclr}};border:1px solid {{row.kclr}}44">{{row.ksym}}</span></td>
            <td>{% if row.entered %}<span class="vbadge" style="background:{{row.uclr}}22;color:{{row.uclr}};border:1px solid {{row.uclr}}44">{{row.usym}}</span>{% else %}<span style="color:#cfd8dc;font-size:11px">--</span>{% endif %}</td>
            <td style="text-align:center">{% if row.match=='y' %}<span style="color:#27ae60;font-size:15px">&#10003;</span>{% elif row.match=='n' %}<span style="color:#e74c3c;font-size:15px">&#10007;</span>{% else %}<span style="color:#cfd8dc">--</span>{% endif %}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </div>
  <div id="t3" class="tp">
    <div style="font-size:18px;font-weight:700;margin-bottom:4px">{{result.genus}}</div>
    <div style="font-size:11px;color:#78909c;background:#f0f4f8;display:inline-block;padding:3px 10px;border-radius:20px;margin-bottom:10px">Enterobacteriaceae</div>
    <div style="font-size:13px;color:#546e7a;line-height:1.7;margin-bottom:12px">{{result.genus_desc}}</div>
    <div class="igrid">
      <div class="ibox"><div class="il">Gram stain</div><div class="iv" style="color:#e74c3c">Negative</div></div>
      <div class="ibox"><div class="il">Cell shape</div><div class="iv">Rod (Bacillus)</div></div>
      <div class="ibox"><div class="il">Metabolism</div><div class="iv">Facultative anaerobe</div></div>
      <div class="ibox"><div class="il">Family</div><div class="iv">Enterobacteriaceae</div></div>
    </div>
    <div style="font-size:12px;font-weight:700;color:#546e7a;margin-bottom:7px">All organisms in {{result.genus}}</div>
    <div class="org-chips">
      {% for o in result.genus_orgs %}<span class="ochip {{'me' if o==result.organism else ''}}">{{o}}</span>{% endfor %}
    </div>
  </div>
  <div id="t4" class="tp">
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:12px">
      <div class="ibox" style="text-align:center"><div style="font-size:20px;font-weight:700;color:#27ae60">{{result.pos_count}}</div><div class="il">Positive tests</div></div>
      <div class="ibox" style="text-align:center"><div style="font-size:20px;font-weight:700;color:#f39c12">{{result.var_count}}</div><div class="il">Variable tests</div></div>
      <div class="ibox" style="text-align:center"><div style="font-size:20px;font-weight:700;color:#e74c3c">{{result.neg_count}}</div><div class="il">Negative tests</div></div>
    </div>
    <div style="font-size:11px;font-weight:700;color:#546e7a;margin-bottom:6px">Key positive tests</div>
    <div class="tag-wrap">{% for t in result.pos_tests %}<span class="tag" style="background:#e8f5e9;color:#2e7d32;border:1px solid #a5d6a7">+ {{t}}</span>{% endfor %}</div>
    <div style="font-size:11px;font-weight:700;color:#546e7a;margin:10px 0 6px">Key negative tests</div>
    <div class="tag-wrap">{% for t in result.neg_tests %}<span class="tag" style="background:#ffebee;color:#c62828;border:1px solid #ef9a9a">- {{t}}</span>{% endfor %}</div>
  </div>
</div>
{% else %}
<div class="card">
  <div class="empty">
    <div style="font-size:44px;margin-bottom:12px">&#127850;</div>
    <div style="font-size:15px;font-weight:700;color:#78909c;margin-bottom:6px">Ready to identify</div>
    <div style="font-size:13px;color:#b0bec5;max-width:340px;margin:0 auto;line-height:1.8">
      Enter biochemical test results on the left and click Identify Organism.
    </div>
  </div>
</div>
{% endif %}
</div>
</div>
<script>
var rowCount={{entered|length}};
function updateCount(){var c=document.querySelectorAll('.trow').length;document.getElementById('cnt').textContent=c;document.getElementById('runbtn').disabled=c<3;}
function quickAdd(name){var ex=document.querySelectorAll('.tname');for(var i=0;i<ex.length;i++) if(ex[i].textContent===name) return;addRowHTML(name,'d');updateCount();}
function addTest(){var name=document.getElementById('newtname').value.trim();var sym=document.getElementById('newtsym').value;if(!name) return;var ex=document.querySelectorAll('.tname');for(var i=0;i<ex.length;i++) if(ex[i].textContent===name) return;addRowHTML(name,sym);document.getElementById('newtname').value='';updateCount();}
function addRowHTML(name,sym){rowCount++;var id='row-'+rowCount;var opts=['+','-','(+)','(-)','d','ND'].map(function(v){return '<option value="'+v+'"'+(v===sym?' selected':'')+'>'+v+'</option>';}).join('');var div=document.createElement('div');div.className='trow'+(sym!=='d'&&sym!=='ND'?' active':'');div.id=id;div.innerHTML='<span class="tname">'+name+'</span><select name="test_'+name+'" class="tsel" onchange="markA(this,\''+id+'\')">'+opts+'</select><button type="button" class="delbtn" onclick="delRow(this)">&#10005;</button><input type="hidden" name="testname_'+rowCount+'" value="'+name+'">';document.getElementById('tgrid').appendChild(div);}
function markA(sel,rid){document.getElementById(rid).classList.toggle('active',sel.value!=='d'&&sel.value!=='ND');}
function delRow(btn){btn.closest('.trow').remove();updateCount();}
function clearAll(){document.getElementById('tgrid').innerHTML='';document.getElementById('cnt').textContent='0';document.getElementById('runbtn').disabled=true;}
function sw(btn,tid){var card=btn.closest('.card');card.querySelectorAll('.tab').forEach(function(b){b.classList.remove('on')});card.querySelectorAll('.tp').forEach(function(p){p.classList.remove('on')});btn.classList.add('on');var el=document.getElementById(tid);if(el) el.classList.add('on');}
function filterTbl(){var q=document.getElementById('psrch').value.toLowerCase();document.querySelectorAll('#ptbl tbody tr').forEach(function(r){r.style.display=r.cells[0].textContent.toLowerCase().includes(q)?'':'none';});}
document.getElementById('newtname').addEventListener('keydown',function(e){if(e.key==='Enter')addTest();});
updateCount();
</script>
</body></html>"""

@app.route("/")
def index():
    return render_template_string(PAGE,
        entered={}, result=None, profile_rows=[], n_entered=0,
        key_tests=KEY_TESTS, summary=SUMMARY, all_tests=ALL_TEST_SUGGESTIONS)

@app.route("/identify", methods=["POST"])
def identify():
    user_tests = {}
    i = 1
    while True:
        tname = request.form.get("testname_" + str(i))
        if not tname:
            break
        tval = request.form.get("test_" + tname, "d")
        user_tests[tname] = tval
        i += 1

    result    = ml_identify(user_tests)
    n_entered = len(user_tests)
    known     = result["profile"]

    profile_rows = []
    for t in test_names:
        ksym = str(known.get(t, "ND"))
        kclr = SYM_CLR.get(ksym, "#95a5a6")
        uv   = user_tests.get(t)
        if uv:
            usym  = uv
            uclr  = SYM_CLR.get(usym, "#95a5a6")
            match = "y" if abs(SYM2NUM.get(ksym,2) - SYM2NUM.get(usym,2)) <= 1 else "n"
            entered = True
        else:
            usym = "--"; uclr = "#cfd8dc"; match = "-"; entered = False
        profile_rows.append({"test":t,"ksym":ksym,"kclr":kclr,
                              "usym":usym,"uclr":uclr,"entered":entered,"match":match})

    prof_data = PROFILES.get(result["organism"], {})
    result["pos_tests"] = prof_data.get("pos_tests", [])
    result["neg_tests"] = prof_data.get("neg_tests", [])
    result["var_tests"] = prof_data.get("var_tests", [])
    result["pos_count"] = len(result["pos_tests"])
    result["neg_count"] = len(result["neg_tests"])
    result["var_count"] = len(result["var_tests"])

    return render_template_string(PAGE,
        entered=user_tests, result=result,
        profile_rows=profile_rows, n_entered=n_entered,
        key_tests=KEY_TESTS, summary=SUMMARY, all_tests=ALL_TEST_SUGGESTIONS)

# THIS IS THE KEY FIX - must bind to PORT environment variable
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
