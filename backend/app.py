from graph.workflow import app

initial_state = {
    "query": "vivo mobiles btw 30000  to 50000"
}

result = app.invoke(initial_state)

print("\n========================")
print("FINAL RESPONSE")
print("========================\n")

print(result["response"])