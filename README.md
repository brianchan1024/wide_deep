## wide_deep train in python and load&predict in java

* install tensorflow in python
* download data with: ```python download_data.py```
* train and save mode with : ```python wide_deep.py```
* show input and output name with: ```saved_model_cli show --dir /SAVED_MODEL_DIR  --all```
* download model in java
* set maven dependency
	```
	<dependency>
            <groupId>com.google.protobuf</groupId>
            <artifactId>protobuf-java</artifactId>
            <version>${protobuf.version}</version>
        </dependency>
        <dependency>
            <groupId>org.tensorflow</groupId>
            <artifactId>tensorflow</artifactId>
            <version>1.6.0</version>
        </dependency>
        <dependency>
            <groupId>org.tensorflow</groupId>
            <artifactId>proto</artifactId>
            <version>1.4.0</version>
        </dependency>
    ```
* predict with java