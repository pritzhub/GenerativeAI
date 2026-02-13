from causalityiq_app.settings import settings

#print(settings)
#print("has attr:", hasattr(settings, "debug_log_llm_prompts"))
print("value:", getattr(settings, "debug_log_llm_prompts", None))