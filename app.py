import gradio as gr
from fastai.vision.all import load_learner, PILImage

# Load your exported model
learn = load_learner('model.pkl')

labels = learn.dls.vocab  # e.g. ['cat', 'dog']

def predict_image(img):
    # Convert the uploaded image to a PIL Image if needed
    pred, pred_idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Pet Classifier"
)

if __name__ == "__main__":
    demo.launch()
