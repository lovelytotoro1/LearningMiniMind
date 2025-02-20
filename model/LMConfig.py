from transformers import PretrainedConfig
"""
    pretrainedconfig 相当于一个 config类别，作者实现 LMConfig 应该是在于利用整个函数定义网络模型的配置，然后后续若是想更改部分参数
    就只需要在本配置文件的基础上修改部分参数，而不是读取默认的pretrainconfig然后全部重新设置

    https://hugging-face.cn/docs/transformers/main_classes/configuration

    基类 PretrainedConfig 实现了从本地文件或目录，或者从库提供的预训练模型配置（从 HuggingFace 的 AWS S3 存储库下载）加载/保存配置的常用方法。
    每个派生配置类都实现了模型特定的属性。所有配置类中都存在一些共同的属性：hidden_size、num_attention_heads 和 num_hidden_layers。文本模型还实现了：vocab_size。
"""
from typing import List


class LMConfig(PretrainedConfig):
    # model_type (str) — 模型类型的标识符，序列化到 JSON 文件中，用于在 AutoConfig 中重新创建正确的对象。

    model_type = "minimind"

    def __init__(
            self,
            dim: int = 512,
            n_layers: int = 8,
            n_heads: int = 8,
            n_kv_heads: int = 2,
            vocab_size: int = 6400,
            hidden_dim: int = None,
            multiple_of: int = 64,
            norm_eps: float = 1e-5,
            max_seq_len: int = 8192,
            rope_theta: int = 1e6,
            dropout: float = 0.0,
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            ####################################################
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: bool = True,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.1,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs,
    ):
        """
        模型基本配置:
        dim: 模型的维度，决定了每个token的嵌入大小。较大的维度可以捕捉更多的信息，但也会增加计算量。
        n_layers: 模型的层数，决定了模型的深度。更多的层数可以增加模型的表达能力，但也可能导致过拟合。
        n_heads: 多头注意力机制中的头数，决定了模型并行处理信息的能力。更多的头数可以捕捉不同子空间的信息。
        n_kv_heads: 用于键值对的头数，通常小于n_heads，可以减少计算量。
        vocab_size: 词汇表的大小，决定了模型可以处理的词汇量。
        hidden_dim: 隐藏层的维度，如果为None，则根据dim和multiple_of计算。
        multiple_of: 隐藏层维度的倍数，确保维度是multiple_of的倍数，通常用于优化计算。
        norm_eps: LayerNorm中的epsilon值，用于数值稳定性，防止除以零。
        max_seq_len: 模型支持的最大序列长度，决定了模型可以处理的序列长度。
        rope_theta: RoPE（Rotary Position Embedding）的theta参数，用于控制位置编码的旋转角度。
        dropout: Dropout率，用于防止过拟合。
        flash_attn: 是否使用Flash Attention机制，加速注意力计算。
        

        MOE代码详解：https://zerolovesea.github.io/2024/04/10/%E6%BA%90%E7%A0%81%E5%88%86%E6%9E%90%EF%BC%9AMoE%E4%B8%93%E5%AE%B6%E6%9E%B6%E6%9E%84/
        混合专家（MoE）配置:
        use_moe: 是否使用混合专家（Mixture of Experts, MoE）机制。MoE机制通过引入多个专家网络来增加模型的表达能力。
        num_experts_per_tok: 每个token选择的专家数量，决定了每个token由多少个专家处理。
        n_routed_experts: 总的专家数量，决定了模型的专家网络的规模。
        n_shared_experts: 是否使用共享专家，共享专家可以减少参数量，但可能降低模型的表达能力。
        scoring_func: 评分函数，默认为'softmax'，用于选择专家。
        aux_loss_alpha: 辅助损失的alpha参数，用于平衡主损失和辅助损失，辅助损失通常用于优化专家选择。
        seq_aux: 是否在序列级别上计算辅助损失，序列级别的辅助损失可以更好地捕捉序列信息。
        norm_topk_prob: 是否标准化top-k概率，用于专家选择，标准化可以确保专家选择的稳定性。
        """
        self.dim = dim  # 模型的维度，即每个token的嵌入维度
        self.n_layers = n_layers  # 模型的层数
        self.n_heads = n_heads  # 多头注意力机制中的头数
        self.n_kv_heads = n_kv_heads  # 用于键值对的头数，通常小于n_heads
        self.vocab_size = vocab_size  # 词汇表的大小
        self.hidden_dim = hidden_dim  # 隐藏层的维度，如果为None，则根据dim和multiple_of计算
        self.multiple_of = multiple_of  # 隐藏层维度的倍数，确保维度是multiple_of的倍数
        self.norm_eps = norm_eps  # LayerNorm中的epsilon值，用于数值稳定性
        self.max_seq_len = max_seq_len  # 模型支持的最大序列长度
        self.rope_theta = rope_theta  # RoPE（Rotary Position Embedding）的theta参数
        self.dropout = dropout  # Dropout率，用于防止过拟合
        self.flash_attn = flash_attn  # 是否使用Flash Attention机制，加速注意力计算
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe  # 是否使用混合专家（Mixture of Experts, MoE）机制
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率
        super().__init__(**kwargs)
