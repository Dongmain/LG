import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Product:
    name: str
    efficacy: str
    ingredients: str
    crowd: str


class KnowledgeEngine:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.products: List[Product] = self._seed_products()
        self.knowledge_docs = self._build_knowledge_docs()
        self.vectorizer = TfidfVectorizer()
        self.doc_matrix = self.vectorizer.fit_transform(self.knowledge_docs)
        self._build_graph()

    @staticmethod
    def _seed_products() -> List[Product]:
        return [
            Product("润清饮A", "滋阴润燥", "麦冬,百合,玉竹", "经常熬夜、口干人群"),
            Product("清和茶B", "清热降火", "金银花,菊花,淡竹叶", "易上火、咽喉不适人群"),
            Product("元气膏C", "健脾益气", "黄芪,党参,茯苓", "疲劳乏力、气虚人群"),
        ]

    def _build_knowledge_docs(self) -> List[str]:
        docs = [
            "口干舌燥 常见于 阴虚 建议 滋阴 润燥 成分 麦冬 百合 玉竹",
            "上火 咽喉不适 常见于 内热偏盛 建议 清热 降火 成分 金银花 菊花",
            "疲劳 乏力 食欲差 常见于 脾气虚 建议 健脾 益气 成分 黄芪 党参 茯苓",
        ]
        for p in self.products:
            docs.append(f"产品 {p.name} 功效 {p.efficacy} 成分 {p.ingredients} 适用人群 {p.crowd}")
        return docs

    def _build_graph(self):
        relations = [
            ("口干舌燥", "对应证型", "阴虚"),
            ("阴虚", "建议功效", "滋阴润燥"),
            ("滋阴润燥", "推荐成分", "麦冬"),
            ("上火", "对应证型", "内热偏盛"),
            ("内热偏盛", "建议功效", "清热降火"),
            ("清热降火", "推荐成分", "金银花"),
            ("疲劳乏力", "对应证型", "脾气虚"),
            ("脾气虚", "建议功效", "健脾益气"),
            ("健脾益气", "推荐成分", "黄芪"),
        ]
        for head, rel, tail in relations:
            self.graph.add_edge(head, tail, relation=rel)

        for p in self.products:
            self.graph.add_edge(p.name, p.efficacy, relation="产品功效")
            for ing in p.ingredients.split(","):
                self.graph.add_edge(p.name, ing.strip(), relation="产品成分")

    def add_product(self, name: str, efficacy: str, ingredients: str, crowd: str):
        self.products.append(Product(name, efficacy, ingredients, crowd))
        self.knowledge_docs.append(f"产品 {name} 功效 {efficacy} 成分 {ingredients} 适用人群 {crowd}")
        self.doc_matrix = self.vectorizer.fit_transform(self.knowledge_docs)
        self.graph.add_edge(name, efficacy, relation="产品功效")
        for ing in ingredients.split(","):
            self.graph.add_edge(name, ing.strip(), relation="产品成分")

    def rag_retrieve(self, query: str, top_k: int = 3) -> List[str]:
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_matrix)[0]
        idx = np.argsort(sims)[::-1][:top_k]
        return [self.knowledge_docs[i] for i in idx]

    def graph_hint(self, query: str) -> List[Tuple[str, str, str]]:
        hints = []
        for node in self.graph.nodes:
            if node in query:
                for succ in self.graph.successors(node):
                    rel = self.graph[node][succ]["relation"]
                    hints.append((node, rel, succ))
        return hints[:5]


class AgentPipeline:
    banned_words = ["治疗", "根治", "绝对", "保证", "最有效", "治愈"]

    def __init__(self, engine: KnowledgeEngine):
        self.engine = engine

    def intent_agent(self, text: str) -> str:
        if any(k in text for k in ["文案", "小红书", "种草"]):
            return "content_generation"
        return "qa"

    def retrieve_agent(self, text: str) -> Dict[str, List]:
        return {
            "rag_docs": self.engine.rag_retrieve(text),
            "graph_hints": self.engine.graph_hint(text),
        }

    def generate_agent(self, text: str, context: Dict[str, List]) -> str:
        rag_text = "\n".join(context["rag_docs"])
        graph_text = "\n".join([f"{h} -[{r}]-> {t}" for h, r, t in context["graph_hints"]])
        if self.intent_agent(text) == "content_generation":
            return (
                "【草本智营·种草文案】\n"
                f"主题：{text}\n\n"
                "最近状态不在线？试试从日常草本调理开始。\n"
                "结合体质特点，选择更匹配的配方，重在日常养护与平衡。\n\n"
                "【知识依据】\n"
                f"{rag_text}\n\n"
                "【图谱关系提示】\n"
                f"{graph_text if graph_text else '暂无直接命中，已基于语义检索生成。'}"
            )
        return (
            "【智能问答回复】\n"
            f"问题：{text}\n\n"
            "可先从作息、饮食和体质调理入手，以下是检索到的相关知识：\n"
            f"{rag_text}\n\n"
            "图谱关系：\n"
            f"{graph_text if graph_text else '暂无直接命中关系。'}\n\n"
            "以上内容仅用于健康科普，不替代医生诊疗建议。"
        )

    def compliance_agent(self, content: str) -> List[str]:
        risks = []
        for w in self.banned_words:
            if re.search(re.escape(w), content):
                risks.append(f"检测到高风险词：{w}")
        if not risks:
            risks.append("未检测到明显高风险医疗宣称词。")
        return risks


@st.cache_resource
def init_system() -> AgentPipeline:
    engine = KnowledgeEngine()
    return AgentPipeline(engine)


def main():
    st.set_page_config(page_title="草本智营 - 智能营销系统", page_icon="🌿", layout="wide")
    st.title("草本智营：中医企业智能营销系统（课程设计MVP）")

    pipeline = init_system()
    engine = pipeline.engine

    tab1, tab2, tab3, tab4 = st.tabs(["产品知识上传", "小红书内容生成", "智能客服问答", "合规检测"])

    with tab1:
        st.subheader("1) 产品知识上传")
        with st.form("product_form"):
            name = st.text_input("产品名称")
            efficacy = st.text_input("产品功效")
            ingredients = st.text_input("产品成分（用英文逗号分隔）")
            crowd = st.text_input("目标人群")
            submitted = st.form_submit_button("上传并结构化存储")
            if submitted:
                if not (name and efficacy and ingredients and crowd):
                    st.warning("请完整填写所有字段。")
                else:
                    engine.add_product(name, efficacy, ingredients, crowd)
                    st.success(f"已上传：{name}")

        df = pd.DataFrame([p.__dict__ for p in engine.products])
        st.dataframe(df, use_container_width=True)

    with tab2:
        st.subheader("2) 一键生成小红书内容")
        topic = st.text_input("输入主题（如：润清饮A + 熬夜口干人群）", key="topic")
        if st.button("生成种草文案"):
            if not topic.strip():
                st.warning("请输入主题。")
            else:
                ctx = pipeline.retrieve_agent(topic)
                content = pipeline.generate_agent(topic + " 小红书种草文案", ctx)
                st.text_area("生成结果", content, height=260)

    with tab3:
        st.subheader("3) 智能问答客服")
        q = st.text_input("请输入用户问题（如：最近容易上火怎么办？）", key="qa")
        if st.button("生成问答回复"):
            if not q.strip():
                st.warning("请输入问题。")
            else:
                ctx = pipeline.retrieve_agent(q)
                answer = pipeline.generate_agent(q, ctx)
                st.text_area("问答结果", answer, height=260)

    with tab4:
        st.subheader("4) 合规检测")
        text = st.text_area("粘贴待检测文本", "本产品可治疗上火并保证见效。", height=180)
        if st.button("执行合规检测"):
            risks = pipeline.compliance_agent(text)
            for item in risks:
                if "高风险" in item:
                    st.error(item)
                else:
                    st.success(item)


if __name__ == "__main__":
    main()
