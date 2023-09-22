"""Python file to serve as the frontend"""
import sys
import os
sys.path.append(os.path.abspath('.'))

from langchain.chat_models.openai import ChatOpenAI

from langchain import PromptTemplate, OpenAI, LLMChain
import chainlit as cl

from chainlit import user_session

from typing import List
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.output_parsers.pydantic import PydanticOutputParser

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.utilities import GoogleSearchAPIWrapper

from langchain.retrievers.web_research import WebResearchRetriever
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# these three lines swap the stdlib sqlite3 lib with the pysqlite3 package
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import re

from langchain.chains import RetrievalQAWithSourcesChain

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain


template = """Question: {question}

Answer: Let's think step by step."""

class LineList(BaseModel):
    """List of questions."""

    lines: List[str] = Field(description="Questions")

class QuestionListOutputParser(PydanticOutputParser):
    """Output parser for a list of numbered questions."""

    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = re.findall(r"\d+\..*?\n", text)
        return LineList(lines=lines)

@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    # Call the chain asynchronously
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # Vectorstore
    vectorstore = Chroma(embedding_function=OpenAIEmbeddings(),persist_directory="./chroma_db_oai")

    # Search
    search = GoogleSearchAPIWrapper()


    # Initialize
    web_research_retriever_llm_chain = WebResearchRetriever(
        vectorstore=vectorstore,
        llm_chain=llm_chain, 
        search=search, 
    )

    lines = res["text"].lines

    docs = web_research_retriever_llm_chain.get_relevant_documents(lines)

    qa_chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="stuff")

    # Create the combine documents chain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt="Let's think step by step.",
    )

    combine_documents_chain 

    # Call RetrievalQAWithSourcesChain
    retrieval_qa_with_sources_chain = RetrievalQAWithSourcesChain(
        retriever=docs.as_retriever(),
        combine_documents_chain=combine_documents_chain,
    )

    content = await retrieval_qa_with_sources_chain.acall(message)

    # Send the response
    await cl.Message(content=content).send()

@cl.on_chat_start
def main():
    user_env = cl.user_session.get("env")
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0), verbose=True)

    llm = ChatOpenAI(temperature=0)

    # LLMChain
    search_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are an assistant tasked with improving Google search 
        results. Generate FIVE Google search queries that are similar to
        this question. The output should be a numbered list of questions and each
        should have a question mark at the end: {question}""",
    )

    llm_chain = LLMChain(
        llm=llm,
        prompt=search_prompt,
        output_parser=QuestionListOutputParser(),
    )

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)

