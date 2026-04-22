"""
Financial LLM Studio  –  Streamlit UI  v2
==========================================
7-tab interface:
  1  Train Spec     – upload/paste/sample spec → engine learns field rules
  2  Validate File  – validate data against any spec
  3  Generate Data  – synthetic test data generation
  4  Auto-Detect    – TF-IDF cosine format identification
  5  Database       – Oracle / PG / MySQL / SQLite connector
  6  Audit Log      – full event history with stats
  7  Spec Browser   – view, export, delete specs

Run:  streamlit run app.py
"""

import os, sys, json, textwrap
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.spec_engine    import SpecEngine
from core.validator      import Validator
from core.generator      import Generator
from core.db_connector   import DBConnector
from core.audit_log      import AuditLog
from formats.builtin_formats import seed_knowledge_base
from formats.swift_mt103     import SWIFT_MT103_SPEC, SwiftMT103Generator

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_data")

# ── Bootstrap ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_engine():
    e = SpecEngine()
    seed_knowledge_base(e.kb)
    e.kb.save("SWIFT_MT103", SWIFT_MT103_SPEC)
    return e

@st.cache_resource
def get_validator():  return Validator(get_engine().kb)
@st.cache_resource
def get_generator():  return Generator(get_engine().kb)
@st.cache_resource
def get_db():         return DBConnector()
@st.cache_resource
def get_audit():      return AuditLog()

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Financial LLM Studio", page_icon="🏦",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');
html,body,[class*="css"]{font-family:'Syne',sans-serif}
[data-testid="stSidebar"]{background:linear-gradient(160deg,#070d1a,#091220);border-right:1px solid #1a2d4a}
[data-testid="stSidebar"] *{color:#a8c4e0!important}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:#58a6ff!important}
.stApp{background:#060d18;color:#c9d1d9}
.stTabs [data-baseweb="tab-list"]{gap:3px;background:#0a1628;padding:6px;border-radius:10px}
.stTabs [data-baseweb="tab"]{background:transparent;border-radius:8px;padding:8px 16px;color:#6e8fab;font-weight:600;font-size:.85rem}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#1f4f8f,#1a6fdb)!important;color:#fff!important}
.out{background:#040b14;border:1px solid #1a2d4a;border-radius:8px;padding:16px;
     font-family:'JetBrains Mono',monospace;font-size:.72rem;color:#7ec8e3;
     max-height:440px;overflow-y:auto;white-space:pre;line-height:1.5}
.vb{display:inline-block;padding:3px 14px;border-radius:20px;font-size:.8rem;font-weight:700}
.ok {background:#0d4429;color:#3fb950;border:1px solid #238636}
.er {background:#4d1818;color:#f85149;border:1px solid #da3633}
.bar-wrap{background:#0a1628;border-radius:6px;height:10px;width:100%;margin:4px 0}
.bar{border-radius:6px;height:10px}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Financial LLM Studio")
    st.caption("Custom Rule-Learning Engine · No External LLM")
    st.markdown("---")
    engine = get_engine()
    specs  = engine.list_specs()
    st.markdown("### 📚 Specs")
    icons  = {"nacha":"🟦","oracle_gl":"🟩","visa_vcf":"🟨","swift_mt103":"🟧","custom":"⬜"}
    for s in sorted(specs):
        sp = engine.get_spec(s) or {}
        ft = sp.get("format_type","")
        st.markdown(f"{icons.get(ft,'⬜')} `{s}`")
    st.markdown("---")
    audit = get_audit()
    ss    = audit.stats()
    st.metric("Specs", len(specs))
    st.metric("Validations", ss.get("validations_run",0))
    st.metric("Generated", ss.get("files_generated",0))
    st.caption("🔒 No external LLM · Rule-based engine")


# ── Tabs ──────────────────────────────────────────────────────────────────
tabs = st.tabs(["🧠  Train","✅  Validate","⚙️  Generate",
                "🔍  Auto-Detect","🗄️  Database","📊  Audit Log","📋  Spec Browser"])
t1,t2,t3,t4,t5,t6,t7 = tabs


# ════════════════════════════════════════════════════════════════════════
# TAB 1 – TRAIN
# ════════════════════════════════════════════════════════════════════════
with t1:
    st.markdown("## 🧠 Train on a Spec Document")
    st.info("Upload or paste any format specification document. The engine extracts "
            "field definitions, data types, lengths, and validation rules — no external LLM needed.")

    ca, cb = st.columns([3,2])
    with ca:
        sname = st.text_input("Spec Name", placeholder="FIRST_NATIONAL_ACH_V3")
        ftype = st.selectbox("Format Family", ["nacha","visa_vcf","oracle_gl","swift_mt103","custom"])
        sdesc = st.text_area("Description", height=60)
        src1  = st.radio("Source", ["📁 Upload File","✏️ Paste Text","📂 Built-in Sample"], horizontal=True)

        stext = ""
        if src1 == "📁 Upload File":
            uf = st.file_uploader("Spec doc", type=["txt","csv","md","json"])
            if uf:
                stext = uf.read().decode("utf-8","replace")
                st.caption(f"{len(stext):,} chars loaded")
        elif src1 == "✏️ Paste Text":
            stext = st.text_area("Spec content", height=280,
                                  placeholder="Field Name  Start  End  Len  Type  Required\n…")
        else:
            samples = {"NACHA ACH":"nacha_spec.txt","Oracle GL":"oracle_gl_spec.txt",
                       "SWIFT MT103":"swift_mt103_spec.txt"}
            sel1 = st.selectbox("Sample spec", list(samples.keys()))
            spath = os.path.join(SAMPLE_DIR, samples[sel1])
            if os.path.exists(spath):
                with open(spath) as fh: stext = fh.read()
                if not sname: sname = sel1.replace(" ","_").upper()
                with st.expander("Preview"):
                    st.text(stext[:600]+"…")
            else:
                st.warning("Sample file not found in sample_data/")

    with cb:
        st.markdown("#### Field definition styles")
        st.code("# Fixed-width table\nFIELD  1-10  10  AN\nAMT   11-22  12  N\n\n"
                "# Positional\n1  RECORD_TYPE  1  N\n2  COMPANY_ID  10  AN\n\n"
                "# Key-value\nROUTING_NUMBER: 9-char ABA routing", language="text")
        st.markdown("#### Format families")
        for ft,desc in [("nacha","94-char fixed-width ACH"),("visa_vcf","VISA Base II"),
                        ("oracle_gl","Pipe-delimited GL journals"),
                        ("swift_mt103","SWIFT tag-based MT103"),("custom","Any format")]:
            st.markdown(f"- `{ft}` — {desc}")

    if st.button("🚀 Train Engine", type="primary", disabled=not (sname and stext.strip())):
        with st.spinner("Learning…"):
            try:
                r = engine.train(sname, stext, ftype, sdesc)
                get_audit().record_training(sname, field_count=r["field_count"], format_type=ftype)
                st.success(f"✅ **{sname}** trained — {r['field_count']} fields / {r['token_count']} tokens")
                c1,c2,c3 = st.columns(3)
                c1.metric("Fields",r["field_count"]); c2.metric("Tokens",r["token_count"])
                c3.metric("Format",r["format_type"].upper())
                if r["fields"]:
                    import pandas as pd
                    df  = pd.DataFrame(r["fields"])
                    col = [c for c in ["name","start","end","length","data_type","required"] if c in df.columns]
                    st.dataframe(df[col], use_container_width=True)
                else:
                    st.warning("No structured fields auto-extracted. The spec is saved — built-in "
                               "formats (NACHA/GL/VCF) have complete field tables.")
                st.rerun()
            except Exception as ex:
                st.error(f"Training failed: {ex}")


# ════════════════════════════════════════════════════════════════════════
# TAB 2 – VALIDATE
# ════════════════════════════════════════════════════════════════════════
with t2:
    st.markdown("## ✅ Validate a Data File")
    specs2 = engine.list_specs()
    if not specs2:
        st.warning("No specs registered. Train a spec first.")
    else:
        col1, col2 = st.columns([2,3])
        with col1:
            vspec = st.selectbox("Spec", sorted(specs2))
            vdelim = st.text_input("Delimiter", "|", help="Blank = fixed-width")
            src2   = st.radio("Data source", ["📁 Upload","✏️ Paste","📂 Sample"], horizontal=True)
            vcont  = ""
            if src2 == "📁 Upload":
                vf = st.file_uploader("Data file", type=["txt","csv","dat","ach"], key="v2")
                if vf: vcont = vf.read().decode("utf-8","replace")
            elif src2 == "✏️ Paste":
                vcont = st.text_area("File content", height=220, key="vp")
            else:
                sf2 = {"NACHA sample":"sample_nacha.ach","Oracle GL sample":"sample_oracle_gl.txt"}
                sv2 = st.selectbox("Sample", list(sf2.keys()))
                sp2 = os.path.join(SAMPLE_DIR, sf2[sv2])
                if os.path.exists(sp2):
                    with open(sp2) as fh: vcont = fh.read()

        with col2:
            if vcont:
                with st.expander("Preview"):
                    st.markdown(f'<div class="out">{"chr(10)".join(vcont.splitlines()[:20])}</div>',
                                unsafe_allow_html=True)
                if st.button("🔍 Validate", type="primary"):
                    with st.spinner("Checking…"):
                        res    = get_validator().validate(vspec, vcont, vdelim or None)
                        report = res.to_dict()
                        get_audit().record_validation(vspec,"UI",is_valid=report["is_valid"],
                            score=report["score"],records=report["total_records"],errors=report["failed"])
                    sc = report["score"]
                    cl = "#3fb950" if report["is_valid"] else "#f85149"
                    st.markdown(
                        f'<span class="vb {"ok" if report["is_valid"] else "er"}">'
                        f'{"✅ VALID" if report["is_valid"] else "❌ INVALID"}</span>',
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f'<div class="bar-wrap"><div class="bar" style="width:{sc:.0f}%;background:{cl}"></div></div>'
                        f'<span style="color:{cl};font-size:1.3rem;font-weight:800">{sc:.1f}%</span>',
                        unsafe_allow_html=True
                    )
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("Records",report["total_records"]); c2.metric("Passed ✅",report["passed"])
                    c3.metric("Failed ❌",report["failed"]); c4.metric("Warnings",len(report["warnings"]))
                    if report["errors"]:
                        st.markdown("#### ❌ Errors")
                        import pandas as pd
                        st.dataframe(pd.DataFrame(report["errors"]),use_container_width=True)
                    if report["warnings"]:
                        with st.expander(f"⚠️ {len(report['warnings'])} warnings"):
                            import pandas as pd
                            st.dataframe(pd.DataFrame(report["warnings"]),use_container_width=True)
                    st.download_button("📥 JSON Report", data=json.dumps(report,indent=2),
                                       file_name=f"validation_{vspec}.json", mime="application/json")
            else:
                st.info("Select a data source.")


# ════════════════════════════════════════════════════════════════════════
# TAB 3 – GENERATE
# ════════════════════════════════════════════════════════════════════════
with t3:
    st.markdown("## ⚙️ Generate Test Data")
    specs3 = engine.list_specs()
    if not specs3:
        st.warning("No specs available.")
    else:
        col1, col2 = st.columns([2,3])
        with col1:
            gspec  = st.selectbox("Format / Spec", sorted(specs3), key="g3")
            gnrows = st.number_input("Detail records", 1, 1000, 5)
            gseed  = st.number_input("Random seed (0=random)", 0, 99999, 42)
            gdb    = st.checkbox("Seed values from Database")
        with col2:
            db_gen = None
            if gdb:
                db3 = get_db()
                if db3.connected:
                    try: gtbls = db3.list_tables()
                    except: gtbls = []
                    if gtbls:
                        gtbl = st.selectbox("Source table", gtbls)
                        if st.button("Load rows"):
                            try:
                                gr = db3.preview_table(gtbl, limit=int(gnrows))
                                st.session_state["db_gen_data"] = gr
                                import pandas as pd; st.dataframe(pd.DataFrame(gr))
                            except Exception as ex: st.error(str(ex))
                else:
                    sp3o = engine.get_spec(gspec) or {}
                    st.info("Not connected — mock data demo available.")
                    if st.button("Load mock DB data"):
                        mr = get_db().get_mock_data(sp3o.get("format_type","custom"), int(gnrows))
                        st.session_state["db_gen_data"] = mr
                        import pandas as pd; st.dataframe(pd.DataFrame(mr))
                db_gen = st.session_state.get("db_gen_data")

        if st.button("⚡ Generate", type="primary"):
            with st.spinner("Generating…"):
                sp3o = engine.get_spec(gspec) or {}
                if sp3o.get("format_type") == "swift_mt103":
                    out3 = SwiftMT103Generator().generate(int(gnrows), seed=int(gseed) or None)
                else:
                    out3 = get_generator().generate(gspec, int(gnrows),
                                                     seed=int(gseed) or None, db_data=db_gen)
                get_audit().record_generation(gspec, rows=int(gnrows),
                                              seed=int(gseed), from_db=bool(db_gen))
            lines3 = out3.splitlines()
            st.success(f"✅ {len(lines3)} lines generated")
            preview3 = "\n".join(lines3[:35])
            st.markdown(f'<div class="out">{preview3}</div>', unsafe_allow_html=True)
            ext3 = {"nacha":"ach","oracle_gl":"txt","visa_vcf":"dat","swift_mt103":"txt"}.get(
                sp3o.get("format_type","custom"), "txt")
            c1, c2 = st.columns(2)
            c1.download_button(f"📥 Download .{ext3}", data=out3,
                               file_name=f"{gspec}_testdata.{ext3}", mime="text/plain")
            c2.download_button("📥 As JSON",
                               data=json.dumps({"spec":gspec,"lines":lines3}),
                               file_name=f"{gspec}_testdata.json", mime="application/json")


# ════════════════════════════════════════════════════════════════════════
# TAB 4 – AUTO-DETECT
# ════════════════════════════════════════════════════════════════════════
with t4:
    st.markdown("## 🔍 Auto-Detect File Format")
    st.info("The engine uses TF-IDF cosine similarity to rank all known specs against an unknown file.")
    col1, col2 = st.columns([2,3])
    with col1:
        src4  = st.radio("Source", ["📁 Upload","✏️ Paste"], horizontal=True, key="ad4")
        samp4 = ""
        if src4 == "📁 Upload":
            af4 = st.file_uploader("Unknown file", type=["txt","csv","dat","ach","json"], key="adf4")
            if af4: samp4 = af4.read().decode("utf-8","replace")
        else:
            samp4 = st.text_area("Paste content", height=280, key="adp4")
    with col2:
        if samp4:
            if st.button("🔍 Identify Format", type="primary"):
                with st.spinner("Analysing…"):
                    matches4 = engine.identify(samp4[:4000])
                if not matches4:
                    st.warning("No specs registered to compare against.")
                else:
                    st.markdown("#### Match Rankings")
                    for rank, m in enumerate(matches4, 1):
                        medal = ["🥇","🥈","🥉"][rank-1] if rank<=3 else f"#{rank}"
                        pct   = min(m["score"]*500, 100)
                        cl    = "#3fb950" if rank==1 else "#8b949e"
                        sp4i  = engine.get_spec(m["spec"]) or {}
                        st.markdown(f"**{medal} {m['spec']}** &nbsp; `{sp4i.get('format_type','?')}` &nbsp; score: `{m['score']:.4f}`")
                        st.markdown(
                            f'<div class="bar-wrap"><div class="bar" style="width:{pct:.1f}%;background:{cl}"></div></div>',
                            unsafe_allow_html=True)
                    best4 = matches4[0]["spec"]
                    st.success(f"🎯 Best match: **{best4}**")
                    if st.button(f"✅ Quick-validate against {best4}"):
                        r4 = get_validator().validate(best4, samp4)
                        rp4 = r4.to_dict()
                        (st.success if rp4["is_valid"] else st.error)(
                            f"{'✅' if rp4['is_valid'] else '❌'} {rp4['score']:.1f}% · {rp4['failed']} errors")
        else:
            st.info("Provide a file to analyse.")
            st.markdown("""
**How it works:**
1. Tokenise the sample file into words and numbers
2. Build a TF-IDF weighted term vector
3. Compute cosine similarity against every known spec's vocabulary
4. Return a ranked list of matching specs

Best results with 1–3 KB of real financial data.
""")


# ════════════════════════════════════════════════════════════════════════
# TAB 5 – DATABASE
# ════════════════════════════════════════════════════════════════════════
with t5:
    st.markdown("## 🗄️ Database Connection")
    db5 = get_db()
    col1, col2 = st.columns([2,3])
    with col1:
        pr5    = st.selectbox("Type", list(DBConnector.PRESETS.keys()),
                               format_func=lambda k: DBConnector.PRESETS[k]["description"])
        cs5    = st.text_input("Connection string",
                                placeholder=DBConnector.PRESETS[pr5]["placeholder"], type="password")
        ca5, cb5 = st.columns(2)
        if ca5.button("🔌 Connect", type="primary", disabled=not cs5):
            try:
                db5.connect(cs5)
                get_audit().record_db_connect(pr5, db5.conn_str or "", True)
                st.success("Connected!"); st.rerun()
            except Exception as ex:
                get_audit().record_db_connect(pr5,"****",False); st.error(str(ex))
        if cb5.button("🔴 Disconnect", disabled=not db5.connected):
            db5.disconnect(); st.rerun()
        st.markdown("---")
        st5 = db5.status
        st.markdown(f"**Status:** {'🟢 Connected' if st5['connected'] else '🔴 Disconnected'}")
        st.markdown(f"**SQLAlchemy:** {'✅' if st5['sqlalchemy'] else '❌'}")
        st.markdown(f"**Oracle driver:** {'✅' if st5['oracle_driver'] else '⚠️ not installed'}")
    with col2:
        if db5.connected:
            sta, stb = st.tabs(["Table Explorer","SQL Query"])
            with sta:
                try: t5tbls = db5.list_tables()
                except: t5tbls = []
                if t5tbls:
                    t5sel = st.selectbox("Table", t5tbls)
                    t5a, t5b = st.columns(2)
                    if t5a.button("📋 Schema"):
                        import pandas as pd
                        st.dataframe(pd.DataFrame(db5.describe_table(t5sel)), use_container_width=True)
                    pn5 = t5b.number_input("Preview rows", 5, 200, 10)
                    if st.button("👁️ Preview"):
                        import pandas as pd
                        rows5 = db5.preview_table(t5sel, int(pn5))
                        st.dataframe(pd.DataFrame(rows5), use_container_width=True)
                        st.session_state["db_gen_data"] = rows5
                        st.info("Rows ready — switch to Generate tab.")
                else:
                    st.info("No tables found.")
            with stb:
                sql5 = st.text_area("SQL", placeholder="SELECT * FROM GL_LINES LIMIT 50", height=110)
                mr5  = st.number_input("Max rows", 1, 5000, 100)
                if st.button("▶️ Run", type="primary"):
                    try:
                        import pandas as pd
                        rows5 = db5.fetch(sql5, max_rows=int(mr5))
                        get_audit().record_db_query(sql5, len(rows5))
                        st.dataframe(pd.DataFrame(rows5), use_container_width=True)
                        st.caption(f"{len(rows5)} rows")
                        st.session_state["db_gen_data"] = rows5
                        st.success("Rows saved — use in Generate tab.")
                    except Exception as ex: st.error(str(ex))
        else:
            st.info("Connect to a database to explore tables and execute queries.")
            st.markdown("""
| Database | Driver | Install |
|---|---|---|
| Oracle (thin) | oracledb | `pip install oracledb` |
| Oracle (legacy) | cx_Oracle | `pip install cx_Oracle` |
| PostgreSQL | psycopg2 | `pip install psycopg2-binary` |
| MySQL | PyMySQL | `pip install pymysql` |
| SQL Server | pyodbc | `pip install pyodbc` |
| SQLite | built-in | — |
""")


# ════════════════════════════════════════════════════════════════════════
# TAB 6 – AUDIT LOG
# ════════════════════════════════════════════════════════════════════════
with t6:
    st.markdown("## 📊 Audit Log")
    audit6 = get_audit()
    s6     = audit6.stats()
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Total Events",   s6.get("total_events",0))
    k2.metric("Specs Trained",  s6.get("specs_trained",0))
    k3.metric("Validations",    s6.get("validations_run",0))
    k4.metric("Files Generated",s6.get("files_generated",0))
    k5.metric("Avg Valid Score",f"{s6.get('avg_valid_score',0)}%")
    st.markdown("---")
    c1,c2,c3 = st.columns([2,2,1])
    af6 = c1.selectbox("Action filter",
                        ["All","TRAIN","VALIDATE","GENERATE","DB_CONNECT","DB_QUERY","DELETE"])
    sf6 = c2.text_input("Spec filter")
    lf6 = c3.number_input("Limit", 10, 500, 50)
    ents6 = audit6.query(action=af6 if af6!="All" else None,
                          spec=sf6 or None, limit=int(lf6))
    if not ents6:
        st.info("No audit entries yet.")
    else:
        import pandas as pd
        df6   = pd.DataFrame(ents6)
        show6 = [c for c in ["ts","action","spec","is_valid","score","records","rows","field_count"] if c in df6.columns]
        if "ts" in df6.columns:
            df6["ts"] = df6["ts"].str[:19].str.replace("T"," ")
        st.dataframe(df6[show6], use_container_width=True, height=400)
        c_dl, c_cl = st.columns(2)
        c_dl.download_button("📥 Export JSON", data=json.dumps(ents6,indent=2,default=str),
                              file_name="audit_log.json", mime="application/json")
        if c_cl.button("🗑️ Clear Log"):
            n6 = audit6.clear()
            st.success(f"Cleared {n6} entries."); st.rerun()


# ════════════════════════════════════════════════════════════════════════
# TAB 7 – SPEC BROWSER
# ════════════════════════════════════════════════════════════════════════
with t7:
    st.markdown("## 📋 Spec Browser")
    specs7 = sorted(engine.list_specs())
    if not specs7:
        st.info("No specs registered.")
    else:
        col1, col2 = st.columns([1,3])
        with col1:
            sel7 = st.radio("Spec", specs7)
        with col2:
            sp7 = engine.get_spec(sel7)
            if sp7:
                c1,c2,c3 = st.columns(3)
                c1.metric("Format",  sp7.get("format_type","?").upper())
                c2.metric("Fields",  len(sp7.get("fields",[])))
                c3.metric("Saved",   sp7.get("_saved_at","built-in")[:10])
                st.markdown(f"**Description:** {sp7.get('description','—')}")
                st.markdown("---")
                vt7, jt7, dt7 = st.tabs(["Fields","Raw JSON","Delete"])
                with vt7:
                    flds7 = sp7.get("fields",[])
                    if flds7:
                        import pandas as pd
                        avail7 = ["name","record_type","start","end","length","data_type","required"]
                        show7  = [c for c in avail7 if any(c in f for f in flds7)]
                        st.dataframe(pd.DataFrame(flds7)[show7], use_container_width=True, height=420)
                    else:
                        st.info("No fields stored.")
                with jt7:
                    st.json(sp7)
                    st.download_button(f"📥 {sel7}.json", data=json.dumps(sp7,indent=2),
                                       file_name=f"{sel7}.json", mime="application/json")
                with dt7:
                    st.warning(f"Permanently delete **{sel7}**?")
                    if st.button("🗑️ Delete", type="secondary"):
                        engine.delete_spec(sel7)
                        get_audit().record_delete(sel7)
                        st.success(f"Deleted {sel7}"); st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#3d5a7a;font-size:.75rem;font-family:JetBrains Mono,monospace'>"
    "Financial LLM Studio v2  ·  Custom Rule-Learning Engine  ·  NACHA · VISA VCF · Oracle GL · SWIFT MT103 · Custom"
    "</p>", unsafe_allow_html=True)
