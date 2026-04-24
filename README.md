## Method

### 预训练

数据采用AFD caption中的抗体蛋白，包括序列和功能描述文本(去掉序列拼接问答得到)。

输入蛋白质序列到冻结的`esm`模型中得到`esm_embedding`，池化后输入`adaptor`将其映射为`pre_embedding`。利用LLM的tokenizer和embedding层处理功能描述文本得到`text_embedding`。  利用infoNCE计算的`pre_embedding`和`text_embedding`的loss，训练`adaptor`。

### LLM微调

利用AFD数据集进行微调，数据为问答形式的文本。

提取instruction,answer,sequence三部分，对于sequence输入esm+adaptor得到`pre_embedding`，instruction和sequence经过LLM的tokenizer+embedding层得到对应的`instruction_embedding`和`answer_embedding`。按照`pre_embedding`+`instruction_embedding`+`answer_embedding`的顺序拼接后加入LLM进行LORA训练。

### Details

1. adaptor接受的输入为`esm_embedding_polled`，输出的`pre_embedding`为(B,L,D)形状，B为batch_size，L固定为100，D和LLM的`hidden_dim`相同(4096)。内部采用Qformer的learned query实现。
2. 预训练阶段计算loss时，将`pre_embedding`池化为(B,D)的形状，`text_embedding`同理，池化后计算inoNCE。由于infoNCE需要正负样本，batch_size必须大于1
3. LLM微调阶段，拼接`pre_embedding`和`instruction_embedding+answer_embedding`在第二维度，得到(B,L1+L2+L3,D)形状的嵌入。
4. 抗体序列包括重链H和轻链L，二者分别输入esm得到esm_embedding，池化后拼接在一起输入adaptor得到pre_embedding。