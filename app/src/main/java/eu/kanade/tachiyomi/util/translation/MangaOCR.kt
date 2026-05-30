package eu.kanade.tachiyomi.util.translation

import ai.onnxruntime.*
import android.content.Context
import android.graphics.Bitmap
import eu.kanade.tachiyomi.util.translation.DataPreProcessingHelper.Companion.convertBitmapToFloatBuffer
import android.util.Log
import java.nio.LongBuffer
import kotlin.system.measureTimeMillis

//@Serializable
//data class PreprocessorConfig(
//    val do_convert_rgb: JsonElement,
//    val do_normalize: Boolean,
//    val do_rescale: Boolean,
//    val do_resize: Boolean,
//    val image_processor_type: String,
//    val image_mean: List<Float>,
//    val image_std: List<Float>,
//    val resample: Int,
//    val rescale_factor: Double,
//    val size: Map<String, Int>,
//    val crop_size: Map<String, Int>? = null
//)
//
//@Serializable
//data class GenerationConfig(
//    val decoder_start_token_id: Int,
//    val early_stopping: Boolean,
//    val eos_token_id: Int,
//    val length_penalty: Float,
//    val max_length: Int,
//    val no_repeat_ngram_size: Int,
//    val num_beams: Int,
//    val pad_token_id: Int,
//    val transformers_version: String
//)
//
//@Serializable
//data class TokenizerConfig(
//    val added_tokens_decoder: Map<String, Map<String, JsonElement>>,
//    val clean_up_tokenization_spaces: Boolean,
//    val cls_token: String,
//    val do_basic_tokenize: Boolean,
//    val do_lower_case: Boolean,
//    val do_subword_tokenize: Boolean,
//    val do_word_tokenize: Boolean,
//    val jumanpp_kwargs: Map<String, String>? = null,
//    val mask_token: String,
//    val mecab_kwargs: Map<String, String>,
//    val model_max_length: Double,
//    val never_split: Boolean? = null,
//    val pad_token: String,
//    val sep_token: String,
//    val strip_accents: String? = null,
//    val subword_tokenizer_type: String,
//    val sudachi_kwargs: Map<String, String>? = null,
//    val tokenize_chinese_chars: Boolean,
//    val tokenizer_class: String,
//    val unk_token: String,
//    val word_tokenizer_type: String
//)

//@Serializable
//data class SpecialTokensMap(
//    val cls_token: Map<String, JsonElement>,
//    val mask_token: Map<String, JsonElement>,
//    val pad_token: Map<String, JsonElement>,
//    val sep_token: Map<String, JsonElement>,
//    val unk_token: Map<String, JsonElement>
//)


class MangaOCR(private val context: Context) {

    private val env = OrtEnvironment.getEnvironment()
    private val encoder: OrtSession
    private val decoder: OrtSession

    private val size = mapOf(
        "width" to 224,
        "height" to 224,
    )

    private val mean = listOf(0.5f, 0.5f, 0.5f)
    private val std = listOf(0.5f, 0.5f, 0.5f)
    private val rescaleFactor = 0.00392156862745098.toFloat()
    private val bosId = 2.toLong()
    private val eosId = 3.toLong()
    private val maxLen = 80
    private var vocab: Map<Int, String>


    private val baseFolder = "manga-ocr/"

    init {
        var elapsed = measureTimeMillis {
            fun loadModel(name: String): OrtSession {
                val bytes = context.assets.open(baseFolder + name).readBytes()
                val options = OrtSession.SessionOptions()
                try {
                    options.addNnapi()
                    Log.d("MangaOCR", "NNAPI enabled")
                } catch (e: Exception) {
                    Log.d("MangaOCR", "NNAPI failed: ${e.message}")
                }
                options.setIntraOpNumThreads(4)
                options.setInterOpNumThreads(4)
                return env.createSession(bytes, options)
            }
            encoder = loadModel("encoder_model.onnx")
            decoder = loadModel("decoder_model.onnx")

            vocab = context.assets.open(baseFolder + "vocab.txt")
                .use { it.bufferedReader().readLines() }
                .mapIndexed { index, token -> index to token }
                .toMap()
        }
        Log.d("MangaOCR", "class init took ${elapsed}ms")


    }

    fun encode(bitmap: Bitmap): Any {
        var hiddenState: Any
        val elapsed = measureTimeMillis {
            val width = size["width"]
            val height = size["height"]

            val floatBuf = convertBitmapToFloatBuffer(bitmap, width!!, height!!, rescaleFactor, mean, std)
            // 2. Run encoder
            val pixelValues = OnnxTensor.createTensor(
                env, floatBuf,
                longArrayOf(1, 3, height.toLong(), width.toLong()),
            )
            val encoderOutputs = encoder.run(mapOf("pixel_values" to pixelValues))
            hiddenState = encoderOutputs["last_hidden_state"]?.get()?.value
                ?: throw RuntimeException("Encoder output not found")
        }
        Log.d("MangaOCR", "Encoding took ${elapsed}ms")
        return hiddenState
    }

    fun decode(hiddenState: Any): String {

        val generatedIds = mutableListOf(bosId)

        val elapsed = measureTimeMillis {
            repeat(maxLen) {
                val seqLen = generatedIds.size.toLong()

                val inputIds = OnnxTensor.createTensor(
                    env,
                    LongBuffer.wrap(generatedIds.toLongArray()),
                    longArrayOf(1, seqLen),
                )

                val encoderTensor = OnnxTensor.createTensor(env, hiddenState)

                val decoderOutputs = decoder.run(
                    mapOf(
                        "input_ids" to inputIds,
                        "encoder_hidden_states" to encoderTensor,
                    ),
                )

                inputIds.close()
                encoderTensor.close()

                val logits = decoderOutputs["logits"]?.get()?.value as? Array<Array<FloatArray>>
                    ?: throw RuntimeException("Decoder output not found")

                decoderOutputs.close()
                // get the last token's logits and pick highest score
                val nextTokenLogits = logits[0][seqLen.toInt() - 1]
                val nextTokenId = nextTokenLogits.indices
                    .maxByOrNull { nextTokenLogits[it] }!!.toLong()

                if (nextTokenId == eosId) return@repeat
                generatedIds.add(nextTokenId)
            }
        }
        Log.d("MangaOCR", "Decoding took ${elapsed}ms")
        val text = untokenize(generatedIds)
        return text
    }

    fun untokenize(tokens: List<Long>): String {
        return tokens
            .asSequence()
            .drop(1)                                    // drop BOS ([CLS] = id 2)
            .filter { it != 0L && it != 3L }            // drop [PAD] and [SEP]
            .mapNotNull { vocab[it.toInt()] }           // id → token string
            .filter { !it.startsWith("[") }             // drop any remaining special tokens
            .joinToString("")                           // join with no spaces (character level)
            .trim()
    }

    fun close() {
        encoder.close()
        decoder.close()
        env.close()
    }
}
