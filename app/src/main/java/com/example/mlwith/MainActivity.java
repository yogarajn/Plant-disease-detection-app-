/*
 *
 */
package com.example.mlwith;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import com.example.mlwith.ml.Model;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    TextView result, confidence;
    ImageView imageView;
    Button picture;
    int imageSize = 224;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        result = findViewById(R.id.result);
        confidence = findViewById(R.id.confidence);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);

        picture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Launch camera if we have permission
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1);
                } else {
                    //Request camera permission if we don't have it.
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
    }

    public void classifyImage(@NonNull Bitmap image) {
        try {
            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);


            byteBuffer.order(ByteOrder.nativeOrder());
            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;

            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++) {
                    int val = intValues[pixel++];
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"Potato healthy", "Potato late blight", "Potato early blight","Error in image"};

            float confidenceThreshold = 0.6f; // Example threshold, adjust as needed
            if (maxConfidence < confidenceThreshold) {
                // If no class has a confidence higher than the threshold, display the fourth message
                result.setText(classes[3]);
            } else {
                // Display the class with the highest confidence
                result.setText(classes[maxPos]);
            }

            StringBuilder s = new StringBuilder(" ");
            for (int i = 0; i < classes.length; i++) {
                s.append(String.format("%s: %1.f%%\n", classes[i], confidences[i] * 100));
            }
            confidence.setText(s.toString());


            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }



    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == 1 && resultCode == RESULT_OK) {
            if (data == null) {
                Log.e("onActivityResult", "Intent data is null");
                Toast.makeText(this, "Failed to retrieve result: data is null", Toast.LENGTH_SHORT).show();
                return;
            }

            Bundle extras = data.getExtras();
            if (extras == null) {
                Log.e("onActivityResult", "Extras are null");
                Toast.makeText(this, "No data available in extras", Toast.LENGTH_SHORT).show();
                return;
            }

            Object dataObject = extras.get("data");
            if (!(dataObject instanceof Bitmap)) {
                Log.e("onActivityResult", "Data is not a Bitmap");
                Toast.makeText(this, "Failed to retrieve image: Data is not a Bitmap", Toast.LENGTH_SHORT).show();
                return;
            }

            Bitmap image = (Bitmap) dataObject;
            if (image == null) {
                Log.e("onActivityResult", "Bitmap is null");
                Toast.makeText(this, "Failed to retrieve image: Bitmap is null", Toast.LENGTH_SHORT).show();
                return;
            }

            try {
                Log.d("onActivityResult", "Original image dimensions: " + image.getWidth() + "x" + image.getHeight());

                int dimension = Math.min(image.getWidth(), image.getHeight());
                Bitmap thumbnail = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                Log.d("onActivityResult", "Thumbnail dimensions: " + thumbnail.getWidth() + "x" + thumbnail.getHeight());

                imageView.setImageBitmap(thumbnail);

                Bitmap scaledImage = Bitmap.createScaledBitmap(thumbnail, imageSize, imageSize, false);
                Log.d("onActivityResult", "Scaled image dimensions: " + scaledImage.getWidth() + "x" + scaledImage.getHeight());

                classifyImage(scaledImage);
            } catch (Exception e) {

            }
        } else {
            Log.e("onActivityResult", "Request code or result code is incorrect");
            Toast.makeText(this, "Failed to retrieve result: incorrect request or result code", Toast.LENGTH_SHORT).show();
        }
    }


}
