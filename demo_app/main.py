"""Python file to serve as the frontend"""
from langchain import OpenAI # LLMMathChain SerpAPIWrapper?
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import initialize_agent, ZeroShotAgent, Tool, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.chains import LLMChain
import os
import chainlit as cl

@cl.on_chat_start
def start():
    memory = cl.user_session.get("memory")
    if memory is None:
        memory = ConversationBufferMemory(memory_key="game_state")

    llm = OpenAI(temperature=0, streaming=True)
    llmc = ChatOpenAI(temperature=0, streaming=True)
    search = GoogleSearchAPIWrapper()

    template = """Summarize the current state of the D&D campaign:
    {game_state}

    The player says:
    {input}
    """

    prompt = PromptTemplate(input_variables=["input", "game_state"], template=template)
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    summry_chain = LLMChain(
        llm=llmc,
        prompt=prompt,
        verbose=True,
        memory=readonlymemory,
    )


    #template = """You are a Dungeon master trying to decide what to say next to the player. You are having the following thoughts based on the current D&D campaign state.  Respond to the player.

    #Dungeon Master:
    #{input}

    #D&D Campaign State:
    #{game_state}
    #"""

    #prompt = PromptTemplate(input_variables=["input", "game_state"], template=template)
    #readonlymemory = ReadOnlySharedMemory(memory=memory)
    #dm_chain = LLMChain(
    #    llm=llmc,
    #    prompt=prompt,
    #    verbose=True,
    #    memory=memory,
    #)

    def reply(input):
        return input

    tools = [
        Tool(
            name="D&D Reference",
            func=search.run,
            description="useful for when you need to answer questions about D&D rules, or look up facts or ideas for reference. You should ask targeted questions.",
        ),
        Tool(
            name="Game State Tracker",
            func=summry_chain.run,
            description="useful for when you summarize the state of the game or campaign.",
        ),
        Tool(
            name="Dungeon Master's Reply",
            func=reply,
            description="useful for when the Dungeon Master is ready to reply to the player.",
            #return_direct=True,
        ),
    ]

    prefix = """Act as a Dungeon Master. Summarize the game state, think about what can happen next in a fun D&D 5e game, get creative, and finally reply to the user. Do not impersonate the player or take more than one speech turn. You have access to the following tools:"""
    suffix = """Reply to the player after their speech turn below.  Your reply should take into account the game state, and take into account anything in the scratchpad.

    Player:
    {input}

    D&D Campaign State:
    {game_state}

    Dungeon Master Scratchpad:
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "game_state", "agent_scratchpad"],
    )

    llm_chain = LLMChain(llm=llmc, prompt=prompt, verbose=True)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        handle_parsing_errors="Check your output and make sure it conforms!",
        max_iterations=10,
        early_stopping_method="generate",
    )

    #agent = initialize_agent(
    #    llm=llmc,
    #    tools=tools,
    #    verbose=True,
    #    memory=memory,
    #)

    cl.user_session.set("agent", agent_chain)
    cl.user_session.set("memory", memory)

@cl.on_message
async def main(message: str):
    agent = cl.user_session.get("agent")
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)

    reply = await cl.make_async(agent.run)(message, callbacks=[cb])

    print(f"Reply: {reply}")


