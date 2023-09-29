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
    llmc = ChatOpenAI(temperature=0, streaming=True, model="gpt-4")
    search = GoogleSearchAPIWrapper()

    template = """The player just said "{input}".  Summarize the updated state of the game after the player's turn.

Campaign Summary: {game_state}
    """

    prompt = PromptTemplate(input_variables=["input", "game_state"], template=template)
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    summry_chain = LLMChain(
        llm=llmc,
        prompt=prompt,
        verbose=True,
        memory=readonlymemory,
    )

    template = """A new D&D is about to begin, get the ball rolling.  As Dungeon Master for this D&D this campaign, what should happen next after a player says "{input}"?

Campaign Summary: {game_state}
    """

    prompt = PromptTemplate(input_variables=["input", "game_state"], template=template)
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    start_chain = LLMChain(
        llm=llmc,
        prompt=prompt,
        verbose=True,
        memory=readonlymemory,
    )

    template = """As Dungeon Master for the current D&D 5e campaign, I need to decide what should happen next after a player says "{input}".

Campaign Summary: {game_state}
    """

    prompt = PromptTemplate(input_variables=["input", "game_state"], template=template)
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    dm_chain = LLMChain(
        llm=llmc,
        prompt=prompt,
        verbose=True,
        memory=readonlymemory,
    )

    template = """As Dungeon Master for the current D&D 5e campaign, what should I reply to a player that just said "{input}"?

Campaign Summary: {game_state}
    """

    prompt = PromptTemplate(input_variables=["input", "game_state"], template=template)
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    reply_chain = LLMChain(
        llm=llmc,
        prompt=prompt,
        verbose=True,
        memory=readonlymemory,
    )

    template = """As Dungeon Master for the current D&D 5e campaign, you should ask for a dice check from the player after they say "{input}".  You may choose to reveal the DC for the roll or not.

Campaign Summary: {game_state}
    """

    prompt = PromptTemplate(input_variables=["input", "game_state"], template=template)
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    roll_chain = LLMChain(
        llm=llmc,
        prompt=prompt,
        verbose=True,
        memory=readonlymemory,
    )

    # Function to roll X dice with Y sides
    def roll_dice(x: int, y: int) -> int:
        return sum([random.randint(1, y) for _ in range(x)])

    tools = [
        Tool(
            name="Dungeons and Dragons Reference",
            func=search.run,
            description="useful for when you need to answer questions about Dungeons and Dragons rules or facts. You should ask targeted questions.",
        ),
        Tool(
            name="Campaign State Tracker",
            func=summry_chain.run,
            description="useful for when you summarize the current state of the game or campaign, when you need to remember what happened in the past, or when you start repeating yourself."
        ),
         Tool(
            name="Campaign Start",
            func=start_chain.run,
            description="useful for when the campaign is just getting started.",
        ),
        Tool(
            name="Dungeon Master Thoughts",
            func=dm_chain.run,
            description="useful for when a Dungeons and Dragons 5e Dungeon Master is thinking about the campaign.",
            #return_direct=True,
        ),
        Tool(
            name="Dungeon Master Reply",
            func=reply_chain.run,
            description="useful for when you, a Dungeons and Dragons 5e Dungeon Master, are deciding how to form a response to the player.",
            return_direct=True,
        ),
        # Tool to roll X dice with Y sides
        #Tool(
        #    name="Dice Roller",
        #    func=roll_dice,
        #    description="useful for when you need to roll X dice with Y sides to do a check in Dungeons and Dragons 5e.",
        Tool(
            name="Dice Check",
            func=roll_chain.run,
            description="useful for when you need to ask the player for a dice check in Dungeons and Dragons 5e.",
        ),
    ]

    prefix = """How would you, a Dungeon Master, reply to the player's message given the current state of the campaign? You have access to the following tools:"""
    suffix = """You are a Dungeon Master speaking with a player summarizing what just happened in the campaign from the player's perspective. Make the story entertaining with lots of twists and turns. Keep the story moving forward. Make sure the player has all the information they need. Always do what the player asks. Reply directly to the player's message, acting out any characters or events that happen too. Don't repeat past responses to the player.

Campaign Summary: {game_state}
Player's Message: {input}
Dungeon Master's Notes:
##############
{agent_scratchpad}
##############
"""

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
        handle_parsing_errors="Check your output and make sure it makes sense as a response to the player's message.  If it doesn't, try again.",
        max_iterations=3,
        #early_stopping_method="generate",
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


