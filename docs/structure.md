graph TD
    %% 定义全局样式
    classDef input fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef stage1 fill:#e8f5e9,stroke:#4caf50,stroke-width:2px;
    classDef stage2 fill:#e3f2fd,stroke:#2196f3,stroke-width:2px;
    classDef stage3 fill:#ffebee,stroke:#f44336,stroke-width:2px;
    classDef decision fill:#fff3e0,stroke:#ff9800,stroke-width:2px;

    %% 输入层
    Q(["🔍 用户查询 (Query Q)"]) ::: input
    Pool(["📄 候选文档池 (总页数 N)"]) ::: input

    %% 阶段一
    subgraph Stage1 [🟢 阶段一：搜索空间感知的弹性粗召回]
        Dispatcher{"N <= 100 ?<br/>(文档长度阈值)"} ::: decision
        BM25["极低算力纯文本检索引擎<br/>(BM25 / BGE-M3单向量)"] ::: stage1
        Bypass["旁路 (Bypass)<br/>跳过粗召回，保障上限"] ::: stage1
    end

    Q --> Dispatcher
    Pool --> Dispatcher
    
    Dispatcher -- "No (海量语料库模式)" --> BM25
    Dispatcher -- "Yes (短文档模式)" --> Bypass
    
    BM25 -- "极速降维压缩" --> Top50["Top-50 候选页<br/>(Candidate Pool)"] ::: input
    Bypass --> Top50

    %% 阶段二
    subgraph Stage2 [🔵 阶段二：重量级视觉精打分]
        ColPali["3B 级别多模态视觉检索模型<br/>(ColPali / ColQwen2.5)"] ::: stage2
        LateInt["多向量晚期交互计算<br/>(Late Interaction)"] ::: stage2
        ColPali --> LateInt
    end

    Top50 ==> ColPali
    Q -.-> ColPali
    LateInt ==> Sv["s_v<br/>(视觉置信度)"] ::: stage2

    %% 阶段三
    subgraph Stage3 [🔴 阶段三：质量感知与动态融合路由]
        direction TB
        St["s_t<br/>(文本置信度)"] ::: stage3
        Qi["q_i<br/>(问题感知特征)"] ::: stage3
        Ri["r_i<br/>(质量感知特征)"] ::: stage3

        Top50 -. "OCR精确匹配" .-> St
        Top50 -. "诊断乱码率/密度" .-> Ri
        Q -. "提取意图" .-> Qi

        Concat["特征拼接单元 (Feature Assembler)<br/>[ s_v, s_t, q_i, r_i ]"] ::: stage3
        
        Sv --> Concat
        St --> Concat
        Qi --> Concat
        Ri --> Concat

        MLP["极轻量两层 MLP 路由器<br/>(动态阻断与权重调配)"] ::: stage3
        Concat ==> MLP
    end

    %% 输出层
    MLP ==> Sfinal(["🏆 终极融合分 (s_final)"]) ::: input
    Sfinal --> Rerank["降序重排 (Rerank)"] ::: input
    Rerank --> Output(["Top-1 / Top-5 候选页<br/>(送入下游 LLM)"]) ::: input