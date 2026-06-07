

import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
import plotly.graph_objects as go
import copy


# ── Page Configuration ────────────────────────────────────────────
st.set_page_config(
   page_title="LLM Fine-Tuning Demo by Suhaani",
   page_icon="⚙️",
   layout="wide"
)


st.title("⚙️ LLM Fine-Tuning Demo by Suhaani")
st.markdown(
   "Fine-tune a language model on your own custom training data — "
   "no coding required! Edit the pairs below, run fine-tuning, and test the results."
)


# ── Session State Initialization ──────────────────────────────────
DEFAULT_TRAINING_DATA = [
   {
       "input": "My order is late",
       "output": "I sincerely apologize for the delay. Could you please share your order ID so I can resolve this immediately?"
   },
   {
       "input": "I want a refund",
       "output": "I understand your concern. Please share your order details and we will process your refund within 3–5 business days."
   },
   {
       "input": "Where is my package",
       "output": "I will help you track your package right away. Could you please share your order number?"
   }
]


for key, val in {
   "training_data": DEFAULT_TRAINING_DATA,
   "training_losses": [],
   "is_fine_tuned": False,
   "ft_model": None,
   "ft_tokenizer": None,
}.items():
   if key not in st.session_state:
       st.session_state[key] = val


# ── Sidebar: Hyperparameters ──────────────────────────────────────
with st.sidebar:
   st.header("⚙️ Hyperparameters")
   st.markdown("Adjust these before fine-tuning.")
  
   epochs = st.slider(
       "Epochs", 1, 10, 3,
       help="How many times the model trains over your entire dataset. More epochs = more learning, but risks overfitting."
   )
  
   learning_rate = st.select_slider(
       "Learning Rate",
       options=[1e-5, 2e-5, 5e-5, 1e-4],
       value=5e-5,
       help="How aggressively weights update per step. Too high = unstable training. Too low = very slow learning."
   )
  
   max_length = st.slider(
       "Max Token Length", 32, 128, 64,
       help="Maximum number of tokens per training example. Longer sequences need more memory."
   )
  
   st.divider()
   st.caption("**Base Model:** DistilGPT-2")
   st.caption("A lightweight GPT-2 variant — great for demos and quick experimentation.")


# ── Section 1: Training Data ──────────────────────────────────────
st.header("📝 Step 1 — Edit Your Training Data")
st.markdown(
   "These **input → output** pairs teach the model how to respond. "
   "Think of each pair as one exam question + correct answer."
)


updated_data = []
for i, pair in enumerate(st.session_state.training_data):
   col_in, col_out, col_del = st.columns([2.5, 3.5, 0.4])
   with col_in:
       inp = st.text_input(
           f"Input {i + 1}", pair["input"], key=f"inp_{i}",
           placeholder="e.g. My order is late"
       )
   with col_out:
       out = st.text_area(
           f"Expected Output {i + 1}", pair["output"], key=f"out_{i}",
           height=72, placeholder="e.g. I'm sorry to hear that..."
       )
   with col_del:
       st.write("")
       st.write("")
       if st.button("✕", key=f"del_{i}", help="Remove this pair"):
           st.session_state.training_data.pop(i)
           st.rerun()
   updated_data.append({"input": inp, "output": out})


st.session_state.training_data = updated_data


btn_add, btn_reset = st.columns([1, 1])
with btn_add:
   if st.button("➕ Add Pair"):
       st.session_state.training_data.append({"input": "", "output": ""})
       st.rerun()


with btn_reset:
   if st.button("🔄 Reset to Defaults"):
       st.session_state.training_data = [d.copy() for d in DEFAULT_TRAINING_DATA]
       st.session_state.training_losses = []
       st.session_state.is_fine_tuned = False
       st.rerun()


# ── Section 2: Fine-Tuning ────────────────────────────────────────
st.header("⚡️ Step 2 — Fine-Tune the Model")
st.markdown(
   "Click the button below to retrain DistilGPT-2 on your training data. "
   "Watch the **loss decrease** — that means the model is learning!"
)


@st.cache_resource
def load_base_model():
   """Load DistilGPT-2 once and cache it for the session."""
   tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
   tokenizer.pad_token = tokenizer.eos_token
   model = GPT2LMHeadModel.from_pretrained("distilgpt2")
   return model, tokenizer


if st.button("▶ Start Fine-Tuning", type="primary"):
   valid_pairs = [
       p for p in st.session_state.training_data
       if p["input"].strip() and p["output"].strip()
   ]
   if not valid_pairs:
       st.error("Please add at least one complete input-output pair before fine-tuning.")
   else:
       with st.spinner("Loading base model — first run may take ~30 seconds..."):
           base_model, tokenizer = load_base_model()
           ft_model = copy.deepcopy(base_model) # keep original untouched
      
       st.success(
           f"✅ Base model loaded. Training on **{len(valid_pairs)} pairs** "
           f"for **{epochs} epoch(s)**..."
       )
      
       # Prepare formatted training texts
       training_texts = [
           f"Input: {p['input']} Output: {p['output']}{tokenizer.eos_token}"
           for p in valid_pairs
       ]
      
       optimizer = AdamW(ft_model.parameters(), lr=learning_rate)
       ft_model.train()
       losses = []
       progress_bar = st.progress(0, text="Initialising training...")
       status_box = st.empty()
      
       for epoch in range(epochs):
           total_loss = 0.0
           for text in training_texts:
               enc = tokenizer(
                   text,
                   return_tensors="pt",
                   max_length=max_length,
                   truncation=True,
                   padding="max_length"
               )
               output = ft_model(**enc, labels=enc["input_ids"])
               loss = output.loss
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()
               total_loss += loss.item()
          
           avg_loss = round(total_loss / len(training_texts), 4)
           losses.append(avg_loss)
           progress_bar.progress((epoch + 1) / epochs, text=f"Epoch {epoch + 1}/{epochs}")
           status_box.info(f"Epoch **{epoch + 1}/{epochs}** complete — Loss: `{avg_loss}`")
      
       st.session_state.training_losses = losses
       st.session_state.ft_model = ft_model
       st.session_state.ft_tokenizer = tokenizer
       st.session_state.is_fine_tuned = True
       st.success("🎉 Fine-tuning complete! Scroll down to test your model.")


# ── Section 3: Training Loss Graph ───────────────────────────────
if st.session_state.training_losses:
   st.header("📊 Training Loss Curve")
   st.markdown(
       "A **downward trend** confirms the model is successfully learning your training data. "
       "The lower the loss, the closer the model's outputs are to your expected responses."
   )
   fig = go.Figure(go.Scatter(
       x=list(range(1, len(st.session_state.training_losses) + 1)),
       y=st.session_state.training_losses,
       mode="lines+markers",
       name="Training Loss",
       line=dict(color="#FF4B4B", width=3),
       marker=dict(size=10, symbol="circle")
   ))
   fig.update_layout(
       xaxis_title="Epoch",
       yaxis_title="Loss",
       title="Training Loss Over Epochs",
       xaxis=dict(tickmode="linear", dtick=1),
       plot_bgcolor="rgba(0,0,0,0)",
       paper_bgcolor="rgba(0,0,0,0)",
       height=340
   )
   st.plotly_chart(fig, use_container_width=True)


# ── Section 4: Test Prompt ────────────────────────────────────────
st.header("🔮 Step 3 — Test Your Fine-Tuned Model")
if st.session_state.is_fine_tuned:
   test_prompt = st.text_input(
       "Enter a test prompt:",
       placeholder="e.g. My order is late"
   )
   if st.button("🚀 Generate Response") and test_prompt:
       with st.spinner("Generating response..."):
           m = st.session_state.ft_model
           t = st.session_state.ft_tokenizer
           m.eval()
           prompt_text = f"Input: {test_prompt} Output:"
           enc = t(prompt_text, return_tensors="pt")
          
           with torch.no_grad():
               output_ids = m.generate(
                   enc["input_ids"],
                   max_new_tokens=60,
                   do_sample=True,
                   temperature=0.7,
                   repetition_penalty=1.3,
                   pad_token_id=t.eos_token_id
               )
           full_text = t.decode(output_ids[0], skip_special_tokens=True)
           response = full_text[len(prompt_text):].strip()
           st.markdown("**Model Response:**")
           if response:
               st.success(response)
           else:
               st.warning(
                   "No output was generated. Try increasing the number of epochs "
                   "or adjusting the learning rate in the sidebar."
               )
else:
   st.info("⬆️ Complete fine-tuning in Step 2 first, then come back here to test.")


# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.caption(
   "🎓 Product Management & AI Course | "
   "Built with [Streamlit](https://streamlit.io) + "
   "[Hugging Face Transformers](https://huggingface.co/docs/transformers)"
)




