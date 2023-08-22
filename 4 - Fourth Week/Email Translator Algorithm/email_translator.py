"""
Author: Isaac Monroy
Title: Email Translator Algorithm
Description:
    The objective of this algorithm is to take as input an email 
    that a user wants to translate, and successfully translate 
    from English to a target language, including German, Hindi or
    Mandarin.
    Pre-trained models are being utilized for this solution due 
    to the fact that they have already been trained on large datasets
    and have a high accuracy on Natural Language Processing and can
    translate sentences and words with precision.
    In addition, the method for translation was to translate each 
    sentence individually for higher accuracy, consistency, and to
    keep the meaning and context of the email.
"""
# Import the MarianMTModel and MarianTokenizer for translation
from transformers import MarianMTModel, MarianTokenizer

# Import tkinter for creating the graphical user interface
import tkinter as tk
# For comobox creation
from tkinter import ttk
# Import messagebox for displaying error messages
from tkinter import messagebox

# Import natural language toolkit, which offers tools for NLP
import nltk
# Download 'punkt' tokenizer model, required by "sent_tokenize"
nltk.download('punkt')
# Import for splitting text into sentences
from nltk.tokenize import sent_tokenize

# Function to translate text using the MarianMT model
def translate(text, target_language):
    """
    Based on the user's selection for a language
    the model shall be loaded and the input text 
    shall be translated from English to either
    German, Hindi or Mandarin.
    """
    # Select the appropriate model for the target language
    if target_language == 'German':
        model_name = 'Helsinki-NLP/opus-mt-en-de'
    elif target_language == 'Hindi':
        model_name = 'Helsinki-NLP/opus-mt-en-hi'
    elif target_language == 'Mandarin':
        model_name = 'Helsinki-NLP/opus-mt-en-zh'
    else:
        raise ValueError('Invalid target language')

    # Load the tokenizer and model
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")
    
    # Generate translated text
    translated_output = model.generate(**inputs)
    translated_text = tokenizer.decode(translated_output[0], skip_special_tokens=True)
    
    return translated_text

# Construct email using provided parts
def create_email(subject, greeting, body, closing, full_name):
    email = """{subject}
{greeting},

{body}

{closing},
{full_name}""".format(subject=subject, greeting=greeting, body=body, closing=closing, full_name=full_name)
    return email

# Translate email parts and display the result in the output_text widget
def translate_email():
    # Get user input from the input fields
    subject = subject_entry.get()
    greeting = greeting_entry.get()
    body = body_text.get("1.0", tk.END).strip()
    closing = closing_entry.get()
    full_name = full_name_entry.get()
    
    # Get the target language from the combobox
    language = language_combobox.get()
    
    try:
        # Translate each email part separately
        translated_subject = translate(subject, language)
        translated_greeting = translate(greeting, language)
        
        # Split the body into sentences and translate each sentence separately
        body_sentences = sent_tokenize(body)
        translated_body_sentences = [translate(sentence, language) for sentence in body_sentences]
        translated_body = " ".join(translated_body_sentences)
        
        translated_closing = translate(closing, language)
        
        translated_email = create_email(translated_subject, translated_greeting, translated_body, translated_closing, full_name)
        
        # Update the output_text widget with the translated email
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, translated_email)
        
        # Update the output_language_label with the target language
        output_language_label.config(text=language)
    except Exception as e:
        # Show an error message if something goes wrong
        messagebox.showerror("Error", str(e))

# Copy translated email to clipboard
def copy_to_clipboard():
    translated_text = output_text.get("1.0", tk.END)
    root.clipboard_clear()
    root.clipboard_append(translated_text)

# Set up the main window
root = tk.Tk()
root.configure(bg="#005288")
root.title("Email Translator")

title = "Email Translator"
instructions_text = """
Instructions for email translation:
    1. Split the email that you are going to translate to each section listed below.
    2. Select the language that you want to translate it to.
    3. Click "Translate" to initiate the process.
    4. Once translated, click "Copy" to copy the email to your clipboard.
"""

# Create input fields for email parts
subject_entry = tk.Entry(root, width=50)
greeting_entry = tk.Entry(root, width=50)
body_text = tk.Text(root, wrap=tk.WORD, height=10)
closing_entry = tk.Entry(root, width=50)
full_name_entry = tk.Entry(root, width=50)

# Title and instructions for user labels
title_label = tk.Label(root, text=title, font=('Georgia', 16, 'bold'), padx=280, pady=10, bg='#A7A9AC')
instructions_label = tk.Label(root, text=instructions_text, justify='left', padx=10, font=('Helvetica', 9, 'bold'), bg="#005288", fg='white')

# Create labels for each email part
subject_label = tk.Label(root, text="Subject:", font=('Helvetica', 10, 'bold'), bg="#005288", fg='white')
greeting_label = tk.Label(root, text="Greeting:", font=('Helvetica', 10, 'bold'), bg="#005288", fg='white')
body_label = tk.Label(root, text="Body:", font=('Helvetica', 10, 'bold'), bg="#005288", fg='white')
closing_label = tk.Label(root, text="Closing:", font=('Helvetica', 10, 'bold'), bg="#005288", fg='white')
full_name_label = tk.Label(root, text="Full Name:", font=('Helvetica', 10, 'bold'), bg="#005288", fg='white')

# Create a dropdown menu for language selection
language_combobox = ttk.Combobox(root, values=["German", "Hindi", "Mandarin"], state="readonly")

# Create buttons for translating and copying the text
translate_button = tk.Button(root, text="Translate", command=translate_email)
copy_button = tk.Button(root, text="Copy", command=copy_to_clipboard)

# Create the output text widget and language label
output_text = tk.Text(root, wrap=tk.WORD, height=10)
output_language_label = tk.Label(root, text="", font=('Helvetica', 12, 'bold'), padx=15, bg="#005288", fg='white')

# Position the widgets on the window
title_label.grid(row=0, column=0, columnspan=2)
instructions_label.grid(row=1, column=0, columnspan=2, sticky="w")
subject_label.grid(row=2, column=0, sticky="w")
subject_entry.grid(row=2, column=1)
greeting_label.grid(row=3, column=0, sticky="w")
greeting_entry.grid(row=3, column=1)
body_label.grid(row=4, column=0, sticky="nw")
body_text.grid(row=4, column=1)
closing_label.grid(row=5, column=0, sticky="w")
closing_entry.grid(row=5, column=1)
full_name_label.grid(row=6, column=0, sticky="w")
full_name_entry.grid(row=6, column=1)
language_combobox.grid(row=7, column=1)
translate_button.grid(row=8, column=1, sticky="w")
copy_button.grid(row=8, column=1, sticky="e")
output_language_label.grid(row=9, column=0, sticky="w")
output_text.grid(row=9, column=1)

# Run the application
root.mainloop()

