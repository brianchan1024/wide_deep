package io.github.brian;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

public class WideDeepDemo {

    public static Feature feature(String... strings) {
        BytesList.Builder b = BytesList.newBuilder();
        for (String s : strings) {
            b.addValue(ByteString.copyFromUtf8(s));
        }
        return Feature.newBuilder().setBytesList(b).build();
    }

    public static Feature feature(float... values) {
        FloatList.Builder b = FloatList.newBuilder();
        for (float v : values) {
            b.addValue(v);
        }
        return Feature.newBuilder().setFloatList(b).build();
    }


    public static void predictInstance() {
        Features features =
                Features.newBuilder()
                        .putFeature("age", feature(30))
                        .putFeature("workclass", feature("State-gov"))
                        .putFeature("fnlwgt", feature(141297f))
                        .putFeature("education", feature("Bachelors"))
                        .putFeature("education_num", feature(13f))
                        .putFeature("marital_status", feature("Married-civ-spouse"))
                        .putFeature("occupation", feature("Prof-specialty"))
                        .putFeature("relationship", feature("Husband"))
                        .putFeature("capital_gain", feature(0f))
                        .putFeature("capital_loss", feature(0f))
                        .putFeature("hours_per_week", feature(40f))
                        .build();
        Example example = Example.newBuilder().setFeatures(features).build();

        try (SavedModelBundle model = SavedModelBundle.load("./model", "serve")) {
            Session session = model.session();
            final String xName = "input_example_tensor";
            final String scoresName = "head/predictions/probabilities:0";

            try (Tensor<String> inputBatch = Tensors.create(new byte[][] {example.toByteArray()});
                 Tensor<Float> output =
                         session
                                 .runner()
                                 .feed(xName, inputBatch)
                                 .fetch(scoresName)
                                 .run()
                                 .get(0)
                                 .expect(Float.class)) {
                System.out.println(Arrays.deepToString(output.copyTo(new float[1][2])));
            }
        }
    }

    public static void main(String[] args) {
        predictInstance();
    }
}
