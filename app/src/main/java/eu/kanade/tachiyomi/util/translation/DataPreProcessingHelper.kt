package eu.kanade.tachiyomi.util.translation

import android.graphics.Bitmap
import androidx.core.graphics.scale
import java.nio.FloatBuffer

class DataPreProcessingHelper {
    companion object {

        fun convertBitmapToFloatBuffer(
            bitmap: Bitmap,
            width: Int,
            height: Int,
            rescaleFactor: Float,
            mean: List<Float>,
            std: List<Float>
        ): FloatBuffer {

            val scaledBitmap = bitmap.scale(width, height)
            val pixels = IntArray(width * height)
            scaledBitmap.getPixels(pixels, 0, width, 0, 0, width, height)
            val floatBuf = FloatBuffer.allocate(1 * 3 * width * height)
            // R channel
            pixels.forEach { px -> floatBuf.put(((px shr 16 and 0xFF) * rescaleFactor - mean[0]) / std[0]) }
// G channel
            pixels.forEach { px -> floatBuf.put(((px shr 8 and 0xFF) * rescaleFactor - mean[1]) / std[1]) }
// B channel
            pixels.forEach { px -> floatBuf.put(((px and 0xFF) * rescaleFactor - mean[2]) / std[2]) }

            floatBuf.rewind()

            return floatBuf
        }
    }
}
