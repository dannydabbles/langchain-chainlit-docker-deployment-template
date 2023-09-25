"""Python file to serve as the frontend"""
from langchain import OpenAI # LLMMathChain SerpAPIWrapper?
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.chains import LLMChain
import os
import chainlit as cl

@cl.on_chat_start
def start():
    llm = OpenAI(temperature=0, streaming=True)
    llmc = ChatOpenAI(temperature=0, streaming=True)
    search = GoogleSearchAPIWrapper()

    template = """This is a transcript of a D&D game:

    {chat_history}

    Write a summary of updated game state after the player's turn:
    {input}
    """

    prompt = PromptTemplate(input_variables=["input", "chat_history"], template=template)
    memory = ConversationBufferMemory(memory_key="chat_history")
    readonlymemory = ReadOnlySharedMemory(memory=memory)

    summry_chain = LLMChain(
        llm=llmc,
        prompt=prompt,
        verbose=True,
        memory=readonlymemory,
    )

    tools = [
        Tool(
            name="Fact Search",
            func=search.run,
            description="useful for when you need to answer questions about D&D rules, or look up facts for reference. You should ask targeted questions",
        ),
        Tool(
            name="Game State Tracker",
            func=summry_chain.run,
            description="useful for when you summarize a conversation. The input to this tool should be a string, representing who will read this summary.",
        ),
    ]

    prefix = """Act as a Dungeon Master keeping game notes, interacting with the player and helping move the story forward. You have access to the following tools:"""
    suffix = """Game History:
    {chat_history}

    Player's Turn:
    {input}

    Scratchpad:
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )

    llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory
    )

    cl.user_session.set("agent", agent_chain)

@cl.on_message
async def main(message: str):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)

    await cl.make_async(agent.run)(message, callbacks=[cb])

