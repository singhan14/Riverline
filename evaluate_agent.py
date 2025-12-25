import os
import uuid
from dotenv import load_dotenv
from langsmith import Client, evaluate
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from agent_graph import agent_graph

# Load environment
load_dotenv()

# 1. Define Dataset
# We simulate a "Gold Dataset" of tricky customer interactions.
# Each item has an input (customer msg) and reference (ideal behavior/outcome).
dataset_name = "Riverline Debt Scenarios"
examples = [
    {
        "inputs": {"message": "I lost my job and I can't pay anything right now. Leave me alone!"},
        "outputs": {"expected_behavior": "Empathetic, non-aggressive, offers time or small payment."}
    },
    {
        "inputs": {"message": "I owe ₹12000. Can I settle for ₹5000?"},
        "outputs": {"expected_behavior": "Reject. Policy for >₹10k is max 20% discount. Offer ₹9600."}
    },
    {
        "inputs": {"message": "I owe ₹5000. I can pay ₹3500 today to close it."},
        "outputs": {"expected_behavior": "Accept. Policy for <₹10k is max 30% discount. ₹3500 is exactly 30% off."}
    },
    {
        "inputs": {"message": "Your company is a scam! I'm reporting you."},
        "outputs": {"expected_behavior": "Calm, professional, de-escalates, provides reassurance."}
    }
]

# 2. Define Evaluators
# We use an LLM to "grade" the agent's response based on the expected behavior.

eval_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def empathy_evaluator(run, example):
    """
    Checks if the agent was empathetic.
    """
    agent_output = run.outputs["messages"][-1].content
    expected = example.outputs["expected_behavior"]
    
    prompt = f"""
    You are a QA auditor for a debt collection agency.
    
    User Input: {example.inputs['message']}
    Agent Response: {agent_output}
    Expected Behavior: {expected}
    
    Rate the Agent's response on Empathy and Policy Adherence.
    Return a score between 0 and 1.
    1 = Perfect empathy and policy.
    0 = Rude or wrong policy.
    
    Return ONLY the number.
    """
    
    result = eval_llm.invoke(prompt).content
    try:
        score = float(result.strip())
        return {"key": "empathy_score", "score": score}
    except:
        return {"key": "empathy_score", "score": 0.5} # Fallback

# 3. Define the Target Function
# This function wraps our Agent Graph to be compatible with LangSmith evaluate.
def target(inputs):
    # Create a unique thread for each test run to ensure clean state
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # Prepend a System Message to enforce robust tool calling
    sys_msg = SystemMessage(content="You are River, a debt resolution agent. Use the provided tools (calculate_emi, check_settlement_policy) carefully. Output valid tool calls only.")
    
    try:
        # Run the graph
        events = agent_graph.stream(
            {"messages": [sys_msg, HumanMessage(content=inputs["message"])]}, 
            config=config, 
            stream_mode="values"
        )
        
        # Get final message
        final_msg = None
        for event in events:
            if "messages" in event:
                final_msg = event["messages"][-1]
                
        if final_msg:
            return {"messages": [final_msg]}
        else:
             return {"messages": [AIMessage(content="Error: No response generated.")]}

    except Exception as e:
        return {"messages": [AIMessage(content=f"Error running agent: {str(e)}")]}

# 4. Run Evaluation
if __name__ == "__main__":
    print(f"🚀 Starting Evaluation on '{dataset_name}'...")
    
    # Create dataset in LangSmith (if not exists, or just use ephemeral examples)
    client = Client()
    
    # Check if dataset exists, if so delete to reset or just append
    if client.has_dataset(dataset_name=dataset_name):
        print(f"Dataset '{dataset_name}' exists. Using existing.")
    else:
        ds = client.create_dataset(dataset_name=dataset_name)
        client.create_examples(
            inputs=[e["inputs"] for e in examples],
            outputs=[e["outputs"] for e in examples],
            dataset_id=ds.id
        )
        print(f"Created dataset '{dataset_name}' with {len(examples)} examples.")

    # Run!
    res = evaluate(
        target,
        data=dataset_name,
        evaluators=[empathy_evaluator],
        experiment_prefix="riverline-tests",
        metadata={"version": "1.0", "agent": "react-debt-negotiator"}
    )
    
    print("\n✅ Evaluation Works! View results in LangSmith UI.")
    print(res)
