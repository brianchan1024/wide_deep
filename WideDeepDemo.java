package io.github.brian;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

public class WideDeepDemo {

    public static void predictInstance() {
        try {
            SavedModelBundle modelBundle = SavedModelBundle.load("./model", "serve");
            Session session = modelBundle.session();
            float[] age_val = {30f};
            byte[][] workclass_val = {"State-gov".getBytes()};
            float[] fnlwgt_val = {141297f};
            byte[][] education_val = {"Bachelors".getBytes()};
            float[] education_num_val = {13f};
            byte[][] marital_status_val = {"Married-civ-spouse".getBytes()};
            byte[][] occupation_val = {"Prof-specialty".getBytes()};
            byte[][] relationship_val = {"Husband".getBytes()};
            float[] capital_gain_val = {0f};
            float[] capital_loss_val = {0f};
            float[] hours_per_week_val = {40f};

            Tensor<Float> age = Tensors.create(age_val);
            Tensor workclass = Tensors.create(workclass_val);
            Tensor<Float> fnlwgt = Tensors.create(fnlwgt_val);
            Tensor education = Tensors.create(education_val);

            Tensor<Float> education_num = Tensors.create(education_num_val);
            Tensor marital_status = Tensors.create(marital_status_val);
            Tensor occupation = Tensors.create(occupation_val);
            Tensor relationship = Tensors.create(relationship_val);
            Tensor<Float> capital_gain = Tensors.create(capital_gain_val);
            Tensor<Float> capital_loss = Tensors.create(capital_loss_val);
            Tensor<Float> hours_per_week = Tensors.create(hours_per_week_val);

            Session.Runner runner = session.runner();
            runner = runner.feed("age", age);
            runner = runner.feed("workclass", workclass);
            runner = runner.feed("fnlwgt", fnlwgt);
            runner = runner.feed("education", education);
            runner = runner.feed("education_num", education_num);
            runner = runner.feed("marital_status", marital_status);
            runner = runner.feed("occupation", occupation);
            runner = runner.feed("relationship", relationship);
            runner = runner.feed("capital_gain", capital_gain);
            runner = runner.feed("capital_loss", capital_loss);
            runner = runner.feed("hours_per_week", hours_per_week);
            Tensor<Float> result = runner.fetch("head/predictions/probabilities:0").run().get(0).expect(Float.class);

            float[][] re = result.copyTo(new float[1][2]);
            System.out.println(re[0][0]);
            System.out.println(re[0][1]);
        } catch (Exception e) {
            System.out.println(e);
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        predictInstance();
    }
}
