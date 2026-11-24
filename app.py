"""
Dashboard phân tích cảm xúc bài đăng Facebook (Streamlit + Plotly).
Chạy:  pip install -r requirements.txt
       streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="Dashboard cảm xúc Facebook", page_icon="📊", layout="wide")

DATA_FILE = Path(__file__).parent / "Facebook_data_txt.csv"
ORDER = ["positive", "neutral", "negative"]
VN = {"positive": "Tích cực (Positive)", "neutral": "Trung lập (Neutral)", "negative": "Tiêu cực (Negative)"}
COLOR = {"positive": "#2E9E5B", "neutral": "#8A94A6", "negative": "#D64545"}
METRICS = {"likes": "Likes", "comments": "Comments", "shares": "Shares"}


# ---------------------------------------------------------------- dữ liệu & mô hình
@st.cache_data(show_spinner=False)
def load_data(src) -> pd.DataFrame:
    df = pd.read_csv(src)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["sentiment"] = df["sentiment"].str.lower().str.strip()
    return df.drop_duplicates("post_id").reset_index(drop=True)


def make_model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )


@st.cache_resource(show_spinner="Đang huấn luyện mô hình dự đoán cảm xúc...")
def train(df: pd.DataFrame):
    """Dự đoán out-of-fold (GroupKFold theo nội dung bài) để tránh rò rỉ do văn bản trùng lặp,
    sau đó huấn luyện mô hình cuối trên toàn bộ dữ liệu để dự đoán bài mới."""
    X, y, groups = df["text"], df["sentiment"], df["text"]
    clf = make_model()
    proba = cross_val_predict(clf, X, y, groups=groups, cv=GroupKFold(n_splits=5), method="predict_proba")
    classes = sorted(y.unique())
    pred = pd.Series([classes[i] for i in proba.argmax(axis=1)], index=df.index)
    conf = pd.Series(proba.max(axis=1), index=df.index)
    final = make_model().fit(X, y)
    return pred, conf, final


# ---------------------------------------------------------------- nạp dữ liệu
st.title("📊 Dashboard phân tích cảm xúc bài đăng Facebook")

if DATA_FILE.exists():
    raw = load_data(DATA_FILE)
else:
    up = st.file_uploader("Tải lên file CSV (cần cột: post_id, text, category, sentiment, likes, comments, shares)")
    if up is None:
        st.stop()
    raw = load_data(up)

pred, conf, model = train(raw)
raw = raw.assign(predicted=pred, confidence=conf, correct=(pred == raw["sentiment"]))

# ---------------------------------------------------------------- bộ lọc (câu d)
with st.sidebar:
    st.header("🔎 Bộ lọc")
    cats = st.multiselect("Danh mục bài đăng", sorted(raw["category"].unique()), default=sorted(raw["category"].unique()))
    sents = st.multiselect("Cảm xúc", ORDER, default=ORDER, format_func=VN.get)
    no_spam = st.checkbox("Loại bài spam", value=False)
    keyword = st.text_input("Tìm theo từ khóa trong nội dung")

df = raw[raw["category"].isin(cats) & raw["sentiment"].isin(sents)]
if no_spam:
    df = df[df["is_spam"] == 0]
if keyword:
    df = df[df["text"].str.contains(keyword, case=False, na=False)]

if df.empty:
    st.warning("Không có bài đăng nào khớp bộ lọc hiện tại.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Tổng quan cảm xúc", "Tương tác", "Danh sách bài đăng", "Đánh giá mô hình"])

# ---------------------------------------------------------------- câu a + b
with tab1:
    counts = df["sentiment"].value_counts().reindex(ORDER, fill_value=0)
    c = st.columns(4)
    c[0].metric("Tổng số bài đăng", f"{len(df):,}", help=f"Toàn bộ dữ liệu: {len(raw):,} bài")
    for col, s in zip(c[1:], ORDER):
        col.metric(VN[s], f"{counts[s]:,}", f"{counts[s] / len(df):.1%}", delta_color="off")

    left, right = st.columns(2)
    dist = counts.rename_axis("sentiment").reset_index(name="Số bài")
    dist["Cảm xúc"] = dist["sentiment"].map(VN)
    cmap = {VN[k]: v for k, v in COLOR.items()}
    left.plotly_chart(px.pie(dist, names="Cảm xúc", values="Số bài", hole=0.45, color="Cảm xúc",
                             color_discrete_map=cmap, title="Tỷ lệ các nhóm cảm xúc"), use_container_width=True)
    right.plotly_chart(px.bar(dist, x="Cảm xúc", y="Số bài", color="Cảm xúc", text="Số bài",
                              color_discrete_map=cmap, title="Số bài đăng theo cảm xúc").update_layout(showlegend=False),
                       use_container_width=True)

    by_cat = df.groupby(["category", "sentiment"]).size().reset_index(name="Số bài")
    by_cat["sentiment"] = by_cat["sentiment"].map(VN)
    st.plotly_chart(px.bar(by_cat, x="category", y="Số bài", color="sentiment", barmode="stack",
                           color_discrete_map=cmap, title="Cảm xúc theo danh mục bài đăng",
                           labels={"category": "Danh mục", "sentiment": "Cảm xúc"}), use_container_width=True)

# ---------------------------------------------------------------- câu c
with tab2:
    agg = df.groupby("sentiment")[list(METRICS)].agg(["sum", "mean"]).reindex(ORDER)
    long = []
    for m, label in METRICS.items():
        for s in ORDER:
            if s in agg.index and not pd.isna(agg.loc[s, (m, "mean")]):
                long.append({"Cảm xúc": VN[s], "Chỉ số": label,
                             "Trung bình/bài": agg.loc[s, (m, "mean")], "Tổng": agg.loc[s, (m, "sum")]})
    long = pd.DataFrame(long)
    a, b = st.columns(2)
    a.plotly_chart(px.bar(long, x="Chỉ số", y="Trung bình/bài", color="Cảm xúc", barmode="group",
                          color_discrete_map=cmap, title="Tương tác trung bình mỗi bài"), use_container_width=True)
    b.plotly_chart(px.bar(long, x="Chỉ số", y="Tổng", color="Cảm xúc", barmode="group",
                          color_discrete_map=cmap, title="Tổng tương tác"), use_container_width=True)

    st.subheader("Bảng thống kê")
    tbl = df.groupby("sentiment")[list(METRICS)].agg(["sum", "mean", "median"]).reindex(ORDER).round(1).dropna(how="all")
    tbl.index = [VN[i] for i in tbl.index]
    tbl.columns = [f"{METRICS[m]} – {dict(sum='Tổng', mean='TB', median='Trung vị')[f]}" for m, f in tbl.columns]
    st.dataframe(tbl, use_container_width=True)
    st.caption("Lưu ý: đây là mối liên hệ mô tả, chưa chứng minh quan hệ nhân quả giữa cảm xúc và mức tương tác.")

# ---------------------------------------------------------------- câu e
with tab3:
    st.caption(f"Hiển thị {len(df):,} bài. Dự đoán được tính theo kiểu out-of-fold (mô hình không thấy bài đó khi huấn luyện).")
    show = df[["post_id", "timestamp", "text", "category", "sentiment", "predicted", "confidence",
               "likes", "comments", "shares"]].rename(columns={
        "post_id": "ID", "timestamp": "Thời gian", "text": "Nội dung", "category": "Danh mục",
        "sentiment": "Cảm xúc thật", "predicted": "Cảm xúc dự đoán", "confidence": "Độ tin cậy"})
    st.dataframe(show, use_container_width=True, height=480, hide_index=True,
                 column_config={"Độ tin cậy": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
    st.download_button("⬇️ Tải danh sách đang lọc (CSV)", show.to_csv(index=False).encode("utf-8-sig"),
                       "bai_dang_da_loc.csv", "text/csv")

    st.subheader("Thử dự đoán cảm xúc cho bài đăng mới")
    txt = st.text_area("Nhập nội dung bài đăng (mô hình học từ văn bản tiếng Anh)", "The Initech app keeps crashing after update.")
    if txt.strip():
        p = pd.Series(model.predict_proba([txt])[0], index=model.classes_).reindex(ORDER)
        st.write(f"**Dự đoán:** {VN[p.idxmax()]} ({p.max():.0%})")
        st.plotly_chart(px.bar(x=[VN[i] for i in p.index], y=p.values, color=[VN[i] for i in p.index],
                               color_discrete_map=cmap, labels={"x": "", "y": "Xác suất"}).update_layout(showlegend=False, height=280),
                        use_container_width=True)

# ---------------------------------------------------------------- đánh giá mô hình
with tab4:
    y, yp = raw["sentiment"], raw["predicted"]
    m = st.columns(3)
    m[0].metric("Accuracy (5-fold)", f"{accuracy_score(y, yp):.1%}")
    m[1].metric("Macro-F1", f"{f1_score(y, yp, average='macro'):.3f}")
    m[2].metric("Baseline (luôn đoán lớp đa số)", f"{(y == y.mode()[0]).mean():.1%}")
    cm = pd.DataFrame(confusion_matrix(y, yp, labels=ORDER), index=ORDER, columns=ORDER)
    st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues", aspect="auto",
                              labels=dict(x="Dự đoán", y="Thực tế", color="Số bài"),
                              title="Ma trận nhầm lẫn"), use_container_width=True)
    conflict = (raw.groupby("text")["sentiment"].nunique() > 1).sum()
    st.info(
        f"**Hạn chế dữ liệu:** chỉ {raw['text'].nunique():,}/{len(raw):,} nội dung là duy nhất, và **{conflict:,} nội dung "
        "xuất hiện với nhiều nhãn cảm xúc khác nhau** ⇒ nhãn có nhiễu, nên độ chính xác bị chặn trên. "
        "Vì vậy mô hình được đánh giá bằng GroupKFold theo nội dung bài để tránh rò rỉ dữ liệu."
    )
