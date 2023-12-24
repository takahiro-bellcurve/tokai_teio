import streamlit as st

st.set_page_config(
    page_title="Tokai Teio",
)

st.write("# Tokai Teio DEMO APP")

st.markdown("""
### 研究用レポジトリ: [GitHub](https://github.com/takahiro-bellcurve/tokai_teio)
衣服の画像検索システムのデモアプリです。
https://tokaiteio.sodashi.cc/search_similar_clothing のページで画像をアップロードし、似た衣服を検索することができます。
""")

st.markdown("### 検索例")
col1, col2 = st.columns(2)
with col1:
    st.write("元画像")
    st.image("public/images/68398320b_b_06_500.jpg", width=200)
with col2:
    st.write("検索結果")
    st.image("public/images/73254137b_17_d_500.jpg", width=200)
