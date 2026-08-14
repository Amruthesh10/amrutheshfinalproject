from graph.workflow import app

initial_state = {
    "query": "Suggest Samsung phone under 20000"
}

result = app.invoke(initial_state)

print("\n========================")
print("FINAL RESPONSE")
print("========================\n")

print(result["response"])