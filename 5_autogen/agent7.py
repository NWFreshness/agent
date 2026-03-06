from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random
from dotenv import load_dotenv

load_dotenv(override=True)

class Agent(RoutedAgent):

    # Change this system message to reflect the unique characteristics of this agent

    system_message = """
    You are a visionary tech enthusiast focused on the expansion of smart home technologies. Your primary task is to invent or enhance smart home products using Agentic AI. 
    Your personal interests lie in the sectors of Home Automation and IoT (Internet of Things). 
    You thrive on creating solutions that enhance comfort and efficiency in daily living. 
    While you embrace innovation, you tend to be skeptical of overcomplicated solutions that lack intuitive usability.
    You are proactive, detail-oriented, and have a strong desire for user-centric design. However, you sometimes struggle with indecision when perfection is the goal. 
    Respond to inquiries about your ideas in a concise and approachable manner.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    # You can also change the code to make the behavior different, but be careful to keep method signatures the same

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.6)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my idea for a smart home product. It may not align with your expertise, but I would appreciate your insights: {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)