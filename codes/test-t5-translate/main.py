def main():
    import ctranslate2
    from sentencepiece import SentencePieceProcessor
    from huggingface_hub import snapshot_download

    model_name = "Heng666/madlad400-3b-ct2-int8"
    model_path = snapshot_download(repo_id=model_name)

    tokenizer = SentencePieceProcessor()
    tokenizer.load(f"{model_path}/sentencepiece.model")
    translator = ctranslate2.Translator(model_path)

    input_text = "I love pizza!"
    input_tokens = tokenizer.encode(f"<2{target_language}> {input_text}", out_type=str)
    results = translator.translate_batch(
        [input_tokens],
        batch_type="tokens",
        max_batch_size=1024,
        beam_size=1,
        no_repeat_ngram_size=1,
        repetition_penalty=2,
    )
    translated_sentence = tokenizer.decode(results[0].hypotheses[0])
    print(translated_sentence)

if __name__ == "__main__":
    main()