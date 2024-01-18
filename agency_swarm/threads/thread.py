import inspect
import time
import logging
from typing import Union

from agency_swarm.agents import Agent
from agency_swarm.messages import MessageOutput
from agency_swarm.user import User
from agency_swarm.util.oai import get_openai_client


class Thread:
    id = None
    api_id: str = None
    thread = None
    run = None

    def __init__(self, agent: Union[Agent, User], recipient_agent: Agent):
        self.agent = agent
        self.recipient_agent = recipient_agent

    @classmethod
    def from_model(cls, thread_model):
        thread = cls.__new__(cls)
        thread.id = thread_model.id
        thread.api_id = thread_model.api_id
        thread.agent = User() if thread_model.agent is None else Agent.from_model(
            thread_model.agent)
        thread.recipient_agent = Agent.from_model(thread_model.recipient_agent)
        return thread

    def get_completion(self, message: str, message_files=None, yield_messages=True, **kwargs):
        client = get_openai_client()
        if not self.thread:
            if self.api_id:
                self.thread = client.beta.threads.retrieve(self.api_id)
            else:
                self.thread = client.beta.threads.create()
                self.api_id = self.thread.id

            # Determine the sender's name based on the agent type
            sender_name = "user" if isinstance(self.agent, User) else self.agent.name
            playground_url = f'https://platform.openai.com/playground?assistant={self.recipient_agent._assistant.id}&mode=assistant&thread={self.thread.id}'
            print(f'THREAD:[ {sender_name} -> {self.recipient_agent.name} ]: URL {playground_url}')

        # Check if a run is active
        if self.run and self.run.status in ['queued', 'in_progress']:
            while self.run.status in ['queued', 'in_progress']:
                print("Detected active run - sleeping")
                time.sleep(0.5)
                self.run = client.beta.threads.runs.retrieve(
                    thread_id=self.thread.id,
                    run_id=self.run.id
                )

        # send message
        client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=message,
            file_ids=message_files if message_files else [],
        )

        if yield_messages:
            yield MessageOutput("text", self.agent.name, self.recipient_agent.name, message)

        # create run
        self.run = client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.recipient_agent.id,
        )

        while True:
            # wait until run completes
            while self.run.status in ['queued', 'in_progress']:
                time.sleep(0.5)
                self.run = client.beta.threads.runs.retrieve(
                    thread_id=self.thread.id,
                    run_id=self.run.id
                )

            # function execution
            if self.run.status == "requires_action":
                tool_calls = self.run.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []
                for tool_call in tool_calls:
                    if yield_messages:
                        yield MessageOutput("function", self.recipient_agent.name, self.agent.name,
                                            str(tool_call.function))

                    agent_output, user_output = self._execute_tool(
                        tool_call, **kwargs)
                    if inspect.isgenerator(agent_output):
                        try:
                            while True:
                                item = next(agent_output)
                                if isinstance(item, MessageOutput) and yield_messages:
                                    yield item
                        except StopIteration as e:
                            agent_output = e.value
                    else:
                        if yield_messages:
                            output = user_output if user_output is not None else agent_output
                            yield MessageOutput("function_output", tool_call.function.name, self.recipient_agent.name, output)

                    tool_outputs.append(
                        {"tool_call_id": tool_call.id, "output": str(agent_output)})

                # submit tool outputs
                self.run = client.beta.threads.runs.submit_tool_outputs(
                    thread_id=self.thread.id,
                    run_id=self.run.id,
                    tool_outputs=tool_outputs
                )
            # error
            elif self.run.status == "failed":
                raise Exception("Run Failed. Error: ", self.run.last_error)
            # return assistant message
            else:
                messages = client.beta.threads.messages.list(
                    thread_id=self.api_id
                )
                message = messages.data[0].content[0].text.value

                if yield_messages:
                    yield MessageOutput("text", self.recipient_agent.name, self.agent.name, message)

                return message

    def get_messages(self):
        if self.api_id:
            client = get_openai_client()
            return client.beta.threads.messages.list(thread_id=self.api_id, order="asc", limit=100)

    def _execute_tool(self, tool_call, **kwargs):
        funcs = self.recipient_agent.functions
        func = next((func for func in funcs if func.__name__ == tool_call.function.name), None)

        if not func:
            return f"Error: Function {tool_call.function.name} not found. Available functions: {[func.__name__ for func in funcs]}"

        try:
            # init tool
            func = func(**eval(tool_call.function.arguments))
            func.caller_agent = self.recipient_agent
            
            # get outputs from the tool
            output = func.run(**kwargs)
            if isinstance(output, tuple):
                return output
            else:
                return output, None
        except Exception as e:
            logging.error(e)
            error_message = f"Error: {e}"
            if "For further information visit" in error_message:
                error_message = error_message.split("For further information visit")[0]
            return error_message, None
