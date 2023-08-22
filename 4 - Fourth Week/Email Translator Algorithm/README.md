# Email Translator Algorithm

## Author
Isaac Monroy

## Project Description
The objective of this algorithm is to take an email as input that a user wants to translate and successfully translate it from English to a target language, including German, Hindi, or Mandarin. Pre-trained MarianMT models are being utilized to ensure high accuracy and precision in the translation.

## Libraries Used
- **transformers**: Importing the MarianMTModel and MarianTokenizer for translation.
- **tkinter**: For creating the graphical user interface and widgets like buttons, labels, etc.
- **nltk**: Provides tools for Natural Language Processing including splitting text into sentences.
- **ttk**: For creating a combobox for language selection.
- **messagebox**: For displaying error messages.

## How to Run:
1. Run the code in your Python environment.
2. Enter the parts of the email in the given fields.
3. Select the target language from the dropdown.
4. Click on the "Translate" button.
5. The translated email will be displayed, and you can copy it to your clipboard by clicking "Copy."

## Input and Output:
- **Input**: Subject, Greeting, Body, Closing, Full Name, Target Language.
- **Output**: Translated email in the selected target language.
