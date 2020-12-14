# from model.Translator import Translator_Factory

def main():
    translator = Translator_Factory().get_translator()
    test_input = "TEXT 2 TRANSLATE"
    translated = translator.translate(test_input)


if __name__ == '__main__':
    main()
