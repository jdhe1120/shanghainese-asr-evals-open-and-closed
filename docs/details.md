# Details

## Reading the results

- **Hearing vs spelling.** On phone conversation CN-MultiDialect-ASR leads folded by over two points but falls behind
  Step-Audio2-Wu-ASR in strict: it hears well but writes MagicData's own spellings (吾, 伐 for veq, 呃), not the teaching
  convention. Step-Audio2-Wu-ASR writes closest to the teaching convention.
- **A different test set changes the order.** CN-MultiDialect-ASR leads phone conversation clearly and only ties for the
  lead on podcasts and storytelling; Conformer-U2pp-Wu moves the other way. Both sets have caveats (‡ in the tables).
- **Check against the paper.** Step-Audio2-Wu-ASR scores 12.9% combined strict CER on our 300-clip sample of
  WenetSpeech-Wu-Bench; the WenetSpeech-Wu paper reports 12.85% on the full set.
- **Closed APIs.** Azure Speech is the only paid API found with a Wu locale (`wuu-CN`); Microsoft doesn't say which
  variety it was trained on. Gemini 3.5 Transcribe offers only Mandarin and Cantonese. Gemini 4 Argon names no speech
  languages; prompted to transcribe, it writes fluent Wu that often isn't what was said.

## Running

- Models: `whisper-wu`, `u2pp-wu`, `cn-multidialect`, `qwen3-asr-stock`, `fun-asr-nano`, `dolphin-cn` (free, local),
  `step-audio2-wu`, `qwen3-omni`, `qwen3-omni-wu` (Modal GPU), `azure-wuu`, `gemini-argon` (paid APIs). `--models all` runs the free ones; paid ones must
  be named. Hugging Face models are pinned to a revision, so a new upload never mixes into old results.
- Every transcription is cached in `cache/<model>/`, keyed on the audio and the reference, so a model is only called for
  clips it hasn't done; paid APIs are never billed twice. `--force` ignores the cache.
- One run at a time. It fills in everything cached, then works one queue: local models one after another (each unloaded
  before the next loads), hosted ones alongside in threads. The page shows one progress bar, a time estimate and a chip per
  model. A second run while one is live is refused: name every model you want in one run.
- `--set wubench` (or `magicdata`, `ssc`) or `--clips` limits what gets transcribed; every other clip still shows its
  cached results. `--model-clips MODEL=a,b` limits one model.
- `scripts/gemini_batch.py gemini-argon submit|status|collect` sends every uncached clip through the Gemini Batch API
  instead (half price, no daily request cap) and writes the results to the cache; batch results have no per-clip time.
  Gemini blocked one ordinary phone clip in the batch (`blockReason: OTHER`); a live retry transcribed it.
- `--cached-only` rebuilds `runs/latest.json` from the cache without calling any model. `python -m shaeval export`
  writes `results/` and the README table.
- On the page, click a score for its diff; shift-click a second score in the same row, or use "compare with", to put two
  models side by side. With no `runs/` (a fresh clone), the page shows the published results rebuilt from
  `results/per_clip.csv`.

### Adding a model

Add `shaeval/models/<name>.py` exposing `transcribe(wav_path) -> str` (optionally `pace()`, called before each request,
and `gap_s()`, its seconds between calls, for rate-limited APIs; or `transcribe_batch(paths)` for a remote GPU), then
register it in `shaeval/models/__init__.py` (`REGISTRY`, `INFO`, `COLUMNS` for its place in the tables, and `PAID` / `CLOSED` if it costs money / is a closed API).

### Bringing your own clips

Put pairs of `<id>.wav` (16 kHz mono) and `<id>.txt` (the reference, UTF-8) in `clips/`, and add the set to
`shaeval/clips.py`.

## Scoring

`shaeval/normalize.py`, `shaeval/metrics.py`: NFKC, digits → Chinese numerals (in the digits views), OpenCC
traditional-to-simplified, then punctuation, symbols and spaces stripped; CER by Levenshtein distance. The headline is the
mean of per-clip CER; `results/summary.csv` also has combined CER (all errors / all reference characters), which weights
long clips more. Four views: strict, spellings folded, digits → 中文, and both (the headline).

- **Strict** rewrites the reference into one teaching spelling first (`rules/wu_canonical.toml`), one character for one,
  at scoring time only, so cached transcripts stay valid. MagicData's 伐 is split into 勿 (veq) and 伐 (the particle va) by
  context.
- **Folded** also counts each "same" pair in `rules/wu_folds.toml` as a match, compared pairwise (never rewritten, so pairs
  don't chain), and accepts the corpus's own character at each position, so the canonical rewrite never makes it worse.
- The author listened to every single-character swap the models made on phone conversation, with the audio, and judged
  which ones are reasonably the same word. Podcasts and storytelling
  was not reviewed separately: it uses the same list, so a spelling variant that only appears there counts as an error.
  Wu spelling isn't standardised, so treat the list as one reasonable reading, not the answer. Rule: the same word spelled differently is folded,
  including when the Mandarin-looking character is the standard Wu spelling (我/吾, 人/拧); a different, Mandarin word (这
  for 搿, 掉 for 脱, 吧 for 伐, 的 for 个) stays an error.

A gap of a point or two between models is within noise at 300 clips; read the diffs.

## Training data

The same test as the Cantonese evals: the share of clips whose output matches the reference exactly, punctuation
included, after NFKC and t2s. Real recognition gets the words right but punctuates its own way; recalled data comes back
verbatim.

- **Phone conversation.** CN-MultiDialect-ASR matches 15% of clips this way, every other model 0-2%. It writes MagicData's
  own spellings (撒 for 啥, 特 for 脱) with commas in the same places, but corrects a known reference typo (巨 for 贵)
  instead of copying it: likely trained on text transcribed MagicData's way, not on these clips.
- **Podcasts & storytelling** has no punctuation in its references, so the test only says "exactly right": ASLP@NPU's four
  models get 10-19% of clips exactly right, every other model 3% or less. The lab built this test set and holds it out of
  training, but their training corpus likely comes from the same kinds of sources.
- The WenetSpeech-Wu authors had the MagicData sample: 27 clips from 9 of its 10 conversations are TTS voice prompts in
  WenetSpeech-Wu-Bench. Whether the conversations are in their 8,000-hour training corpus can't be checked: that corpus
  isn't public.

## Known issues and gotchas

- `pip install dolphin` is an unrelated package; Dolphin-CN-Dialect is `dataoceanai-dolphin` (import `dolphin`). Its own
  download asks ModelScope for a misspelled repo; `scripts/fetch_models.py` fetches the right one.
- **WeNet releases need fixes to load** (applied by `scripts/fetch_models.py`): configs missing `input_dim` /
  `output_dim`, a vocabulary under two names, and training-time fbank dither left on in Conformer-U2pp-Wu, which made its
  output random. Dolphin applies dither in its own code, so `dolphin_cn.py` re-seeds torch per clip.
- WeNet and CN-MultiDialect-ASR (original Qwen3-ASR format, `qwen-asr`, which pins transformers 4.57.6) run in isolated
  `uv run` workers; the main env needs transformers >= 5.17 for Fun-ASR-Nano. Workers point stdout at stderr before
  importing anything: WeNet prints at import, which once desynced the replies.
- Qwen3-ASR has no Wu language setting; it runs with auto language detection. Fun-ASR-Nano runs as `zh`.
- MagicData references contain some errors (e.g. 巨 for 贵); they are scored as they are.

## Test data

### Podcasts & storytelling (WenetSpeech-Wu-Bench)

`scripts/prepare_wubench.py` downloads the ASR test set (`understanding/asr.parquet`) from
[ASLP-lab/WenetSpeech-Wu-Bench](https://huggingface.co/datasets/ASLP-lab/WenetSpeech-Wu-Bench), pinned to a revision.
Of its 4,851 utterances, 4,579 are 1.5-20 s with at least 4 characters and a label not seen before; 300 of those are drawn
with a fixed seed.

- **Not only Shanghainese.** The set has no sub-dialect tag: some clips are Suzhounese or other Wu, and some are Mandarin.
- **Mandarin spellings in the references.** Some references write a Mandarin form where the speaker says the Wu word
  (这 for 搿, 的 for 个), so a model that writes the Wu word correctly can lose characters.
- **No punctuation in the references**, which limits the training-data test (see above).

### Phone conversation (MagicData sample)

`scripts/prepare_magicdata.py` cuts the free sample of MagicData's Shanghai Dialect Conversational Speech Corpus: 10 phone
conversations, two Shanghai speakers each, aged 47-69, 4,546 timestamped segments. A segment is kept if it is one speaker,
with no overlap and no unintelligible, personal-information or filler tags, 1.5-20 s and at least 4 characters; noise tags
are stripped from the reference. 30 per conversation are drawn with a fixed seed. Clip ids are the line numbers in the
corpus's transcript, so they stay stable. The references use their own Wu spellings (吾, 伐 for both 勿 and the particle,
呃 for 个) and contain some errors (巨 for 贵). The licence forbids redistribution, so `results/` has scores only.

### One reader (Shanghai Spoken Corpus sentences, in progress)

The author reads sentences from the University of Alberta's
[Shanghai Spoken Corpus](https://sites.ualberta.ca/~johnnewm/SC/Shanghai/About.html) aloud. The audio is one person's voice, so it is [on Hugging Face](https://huggingface.co/datasets/jdhe1120/shanghainese-ssc-read-speech) for evaluation only, behind an access
request (no training, voice cloning or redistribution); `scripts/fetch_ssc.py` puts it in `clips/`. Its references and
every model's transcript are in `results/per_clip.csv`. The reference is the corpus sentence, or the reader's edit of it where they said it differently. No
model can have trained on this audio. It is left out of the tables until it has 100 clips; read speech by one speaker is
easier than conversation, so compare models within the set.
