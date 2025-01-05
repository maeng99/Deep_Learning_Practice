class MyConv2D(Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='SAME', activation=None, **kwargs):
        super(MyConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.filters),
            initializer='glorot_uniform',
            trainable=True,
            name='kernel'
        )
        self.b = self.add_weight(
            shape=(self.filters,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
        super(MyConv2D, self).build(input_shape)  # 필수

    def call(self, inputs):
        # 컨볼루션 연산 수행
        x = tf.nn.conv2d(inputs, self.w, strides=[1, *self.strides, 1], padding=self.padding)

        # 편향 추가
        x = tf.nn.bias_add(x, self.b)

        # 활성화 함수 적용
        if self.activation is not None:
            x = self.activation(x)
        return x

    def get_config(self):
        config = super(MyConv2D, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': tf.keras.activations.serialize(self.activation)
        })
        return config