package eu.kanade.tachiyomi.util.translation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.japanese.JapaneseTextRecognizerOptions
import kotlin.math.hypot

fun Rect.vertical(): Boolean {
    return this.height() > this.width()
}

fun Rect.distanceToCenterOf(other: Rect): Double {
    // Calculate the center (X, Y) of the first rectangle
    val center1X = this.exactCenterX()
    val center1Y = this.exactCenterY()

    // Calculate the center (X, Y) of the second rectangle
    val center2X = other.exactCenterX()
    val center2Y = other.exactCenterY()

    // Compute the differences
    val dx = (center2X - center1X).toDouble()
    val dy = (center2Y - center1Y).toDouble()

    // Return the hypotenuse: sqrt(dx^2 + dy^2)
    return hypot(dx, dy)
}

class Box {
    var rect = Rect(Int.MAX_VALUE, Int.MAX_VALUE, Int.MIN_VALUE, Int.MIN_VALUE)
    var length = 0
    var textBlocks = mutableListOf<Text.TextBlock>()

    fun mergeBlock(candidate: Text.TextBlock) {
        textBlocks.add(candidate)

        for (point in candidate.cornerPoints!!) {
            rect.left = minOf(rect.left, point.x)
            rect.top = minOf(rect.top, point.y)
            rect.right = maxOf(rect.right, point.x)
            rect.bottom = maxOf(rect.bottom, point.y)
        }
        length++
    }

    fun outset(pixels: Int) {
        // scales out borders by a radius for bigger capture group
        rect.inset(-pixels, -pixels)
    }

    fun isCloseTo(candidate: Rect): Boolean {
        // current rect is close to candidate if
        // candidate.minx - rect.maxx .abs is less < threshold and y bounds are close
        // OR
        // rect.minx - candidate.maxx .abs is less < threshold and y bounds are close
        // OR
        // candidate.miny - rect.maxy .abs is less < threshold and x bounds are close
        // OR
        // rect.miny - candidate.maxy .abs is less < threshold and x bounds are close
//
//        val gapThreshold = 10
//        val alignThreshold = 40
        val distanceThreshold = 40

        // if orientation or shape of rects are off skip them
        if (rect.vertical() != candidate.vertical()) return false

        if (rect.contains(candidate)) return true

        return rect.distanceToCenterOf(candidate) < distanceThreshold
    }
}

class TextDetector(private val context: Context) {
    private val recognizer: TextRecognizer =
        TextRecognition.getClient(JapaneseTextRecognizerOptions.Builder().build())

    fun detect(bitmap: Bitmap): List<Box> {
        val image = InputImage.fromBitmap(bitmap, 0)
        val result = Tasks.await(recognizer.process(image))
        return mergeBoundingBoxes(result.textBlocks)
    }

    fun mergeBoundingBoxes(boxes: List<Text.TextBlock>): List<Box> {
        // dfs
        // start at one bounding box and calculate its low x, high, low y, high y.
        // set bounds of the group to be that boxes bounds
        // check next box and compare its high x to curr low x, etc and if closer than 20 pixels,
        // group them together, and change the groups bounds such that the y highs are the max of the two, and the
        // y lows are the mins of the two, etc
        // continue this until all boxes are explored.

        val visited = mutableSetOf<Text.TextBlock>()
        val stack = ArrayDeque<Text.TextBlock>()

        val mergedBoxes = mutableListOf<Box>()

        for (box in boxes) {
            if (visited.contains(box)) continue

            stack.clear()
            stack.addLast(box)
            visited.add(box)
            val currBox = Box()

            while (!stack.isEmpty() && currBox.length < 6) {
                // pop current bounding box and merge it with the current merged cumulative box
                // and mark as visited dont merge more than 6 boxes
                val curr = stack.removeLast()
                currBox.mergeBlock(curr)

                // check other boxes for proximity and if the pass the check push to the stack
                // so that they get merged to current merged bounding box
                for (box2 in boxes) {
                    if (visited.contains(box2)) continue

                    if (currBox.isCloseTo(box2.boundingBox!!)) {
                        visited.add(box2)
                        stack.addLast(box2)
                    }
                }
            }
            // scale mergedbox outward by 6 pixels for larger capture range and add it
            currBox.outset(3)
            mergedBoxes.add(currBox)
        }

        return mergedBoxes.toList()

    }
}
