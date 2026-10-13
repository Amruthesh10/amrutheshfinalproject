from graph.workflow import app

initial_state = {
    "query": "Suggest  phone under 20000 with good battery backup"
}

result = app.invoke(initial_state)

print("\n========================")
print("FINAL RESPONSE")
print("========================\n")

print(result["response"])