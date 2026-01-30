/**
 * Shader 代码语法高亮
 * 针对 HLSL/GLSL 语法的高亮显示
 * 
 * 使用方法：
 * 1. 在 HTML 中使用 <pre class="shader-code"> 包裹代码
 * 2. 页面加载后会自动高亮
 * 3. 也支持 <div class="code-block" data-lang="hlsl"> 的形式
 */

const ShaderHighlight = {
    // HLSL/GLSL 关键字
    keywords: [
        // 类型
        'void', 'bool', 'int', 'uint', 'float', 'half', 'fixed', 'double',
        'float2', 'float3', 'float4', 'half2', 'half3', 'half4', 'fixed2', 'fixed3', 'fixed4',
        'int2', 'int3', 'int4', 'uint2', 'uint3', 'uint4',
        'float2x2', 'float3x3', 'float4x4', 'half2x2', 'half3x3', 'half4x4',
        'sampler2D', 'sampler3D', 'samplerCUBE', 'sampler', 'Texture2D', 'SamplerState',
        'Varyings', 'Attributes', 'VertexPositionInputs', 'VertexNormalInputs',
        
        // 流程控制
        'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default', 'break', 'continue', 'return', 'discard',
        
        // 结构
        'struct', 'cbuffer', 'tbuffer', 'in', 'out', 'inout', 'uniform', 'static', 'const',
        
        // Unity Shader 结构关键字
        'Shader', 'SubShader', 'Pass', 'Tags', 'Properties', 'HLSLPROGRAM', 'ENDHLSL', 'HLSLINCLUDE',
        'CGPROGRAM', 'ENDCG', 'CGINCLUDE', 'Fallback', 'CustomEditor', 'Queue', 'RenderType',
        'LightMode', 'Blend', 'ZWrite', 'ZTest', 'Cull', 'ColorMask', 'Offset', 'Stencil',
        
        // 预处理指令
        'pragma', 'include', 'define', 'ifdef', 'ifndef', 'endif', 'if', 'else', 'elif', 'undef',
        'vertex', 'fragment', 'geometry', 'hull', 'domain', 'target', 'multi_compile', 'shader_feature'
    ],

    // 内置函数
    builtinFunctions: [
        // 数学函数
        'abs', 'acos', 'all', 'any', 'asin', 'atan', 'atan2', 'ceil', 'clamp', 'clip',
        'cos', 'cosh', 'cross', 'degrees', 'determinant', 'distance', 'dot', 'exp', 'exp2',
        'faceforward', 'floor', 'fmod', 'frac', 'frexp', 'fwidth', 'isfinite', 'isinf', 'isnan',
        'ldexp', 'length', 'lerp', 'log', 'log10', 'log2', 'max', 'min', 'modf', 'mul',
        'normalize', 'pow', 'radians', 'reflect', 'refract', 'round', 'rsqrt', 'saturate',
        'sign', 'sin', 'sincos', 'sinh', 'smoothstep', 'sqrt', 'step', 'tan', 'tanh',
        'transpose', 'trunc',
        
        // 纹理采样
        'tex2D', 'tex2Dlod', 'tex2Dproj', 'texCUBE', 'SAMPLE_TEXTURE2D', 'SAMPLE_TEXTURE2D_LOD',
        'LOAD_TEXTURE2D',
        
        // Unity URP 函数
        'TransformObjectToWorld', 'TransformWorldToObject', 'TransformWorldToView',
        'TransformWorldToHClip', 'TransformObjectToHClip', 'TransformViewToHClip',
        'GetVertexPositionInputs', 'GetVertexNormalInputs',
        'ComputeScreenPos', 'ComputeFogFactor', 'MixFog',
        'SampleSH', 'SampleSHVertex', 'GetMainLight', 'GetAdditionalLight',
        
        // 数据打包
        'PackNormalOctRectEncode', 'UnpackNormalOctRectEncode'
    ],

    // 语义
    semantics: [
        'POSITION', 'NORMAL', 'TANGENT', 'COLOR', 'COLOR0', 'COLOR1',
        'TEXCOORD0', 'TEXCOORD1', 'TEXCOORD2', 'TEXCOORD3', 'TEXCOORD4', 'TEXCOORD5', 'TEXCOORD6', 'TEXCOORD7',
        'SV_POSITION', 'SV_Target', 'SV_Target0', 'SV_Target1', 'SV_Target2', 'SV_Target3',
        'SV_Depth', 'SV_VertexID', 'SV_InstanceID', 'SV_IsFrontFace', 'SV_PrimitiveID',
        'VFACE', 'VPOS'
    ],

    // Unity 内置变量
    unityVariables: [
        '_Time', '_SinTime', '_CosTime', 'unity_DeltaTime',
        '_ProjectionParams', '_ScreenParams', '_ZBufferParams',
        'unity_ObjectToWorld', 'unity_WorldToObject', 'unity_MatrixV', 'unity_MatrixVP',
        'UNITY_MATRIX_M', 'UNITY_MATRIX_V', 'UNITY_MATRIX_P', 'UNITY_MATRIX_VP', 'UNITY_MATRIX_MV', 'UNITY_MATRIX_MVP',
        '_WorldSpaceCameraPos', '_WorldSpaceLightPos0',
        '_LightColor0', '_MainTex', '_MainTex_ST', '_Color', '_BaseMap', '_BaseColor'
    ],

    /**
     * 高亮代码
     * @param {string} code - 原始代码
     * @returns {string} - 带有高亮标记的 HTML
     */
    highlight: function(code) {
        // 先保护字符串和注释
        const stringPlaceholders = [];
        const commentPlaceholders = [];

        // 保护多行注释
        code = code.replace(/\/\*[\s\S]*?\*\//g, (match) => {
            const placeholder = `__MLCOMMENT_${commentPlaceholders.length}__`;
            commentPlaceholders.push({ placeholder, content: match, type: 'comment' });
            return placeholder;
        });

        // 保护单行注释
        code = code.replace(/\/\/.*$/gm, (match) => {
            const placeholder = `__SLCOMMENT_${commentPlaceholders.length}__`;
            commentPlaceholders.push({ placeholder, content: match, type: 'comment' });
            return placeholder;
        });

        // 保护字符串
        code = code.replace(/"[^"]*"/g, (match) => {
            const placeholder = `__STRING_${stringPlaceholders.length}__`;
            stringPlaceholders.push({ placeholder, content: match });
            return placeholder;
        });

        // 高亮语义（需要在关键字之前，因为语义更具体）
        this.semantics.forEach(semantic => {
            const regex = new RegExp(`\\b(${semantic})\\b`, 'g');
            code = code.replace(regex, '<span class="hlsl-semantic">$1</span>');
        });

        // 高亮 Unity 变量
        this.unityVariables.forEach(variable => {
            const regex = new RegExp(`\\b(${variable.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})\\b`, 'g');
            code = code.replace(regex, '<span class="hlsl-unity-var">$1</span>');
        });

        // 高亮内置函数
        this.builtinFunctions.forEach(func => {
            const regex = new RegExp(`\\b(${func})\\s*(?=\\()`, 'g');
            code = code.replace(regex, '<span class="hlsl-builtin">$1</span>');
        });

        // 高亮关键字
        this.keywords.forEach(keyword => {
            const escaped = keyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const regex = new RegExp(`\\b(${escaped})\\b`, 'g');
            code = code.replace(regex, '<span class="hlsl-keyword">$1</span>');
        });

        // 高亮数字
        code = code.replace(/\b(\d+\.?\d*f?)\b/g, '<span class="hlsl-number">$1</span>');

        // 高亮预处理指令（#pragma, #include 等）
        code = code.replace(/(#\w+)/g, '<span class="hlsl-preprocessor">$1</span>');

        // 恢复字符串
        stringPlaceholders.forEach(item => {
            code = code.replace(item.placeholder, `<span class="hlsl-string">${escapeHtml(item.content)}</span>`);
        });

        // 恢复注释
        commentPlaceholders.forEach(item => {
            code = code.replace(item.placeholder, `<span class="hlsl-comment">${escapeHtml(item.content)}</span>`);
        });

        return code;
    },

    /**
     * 初始化页面上所有的代码块
     */
    initCodeBlocks: function() {
        // 处理 .shader-code 类的代码块
        document.querySelectorAll('.shader-code').forEach(block => {
            if (block.dataset.highlighted === 'true') return;
            
            const code = block.textContent;
            block.innerHTML = this.highlight(code);
            block.dataset.highlighted = 'true';
        });

        // 也处理带有 language-hlsl 类的 pre > code 块
        document.querySelectorAll('pre code.language-hlsl, pre code.language-shader').forEach(block => {
            if (block.dataset.highlighted === 'true') return;
            
            const code = block.textContent;
            block.innerHTML = this.highlight(code);
            block.dataset.highlighted = 'true';
        });

        // 处理所有 .code-block 元素（如果还没有高亮）
        // 检查是否包含已经高亮的 span 标签，如果有则跳过
        document.querySelectorAll('.code-block').forEach(block => {
            if (block.dataset.highlighted === 'true') return;
            
            // 检查是否已经包含高亮的 span 标签
            const hasHighlightedSpans = block.querySelector('span.hlsl-keyword, span.hlsl-comment, span.hlsl-builtin, span.keyword, span.comment, span.function');
            if (hasHighlightedSpans) {
                // 已经手动高亮过了，标记为已处理
                block.dataset.highlighted = 'true';
                return;
            }
            
            const code = block.textContent;
            // 只有当内容看起来像代码时才高亮
            if (this.looksLikeCode(code)) {
                block.innerHTML = this.highlight(code);
            }
            block.dataset.highlighted = 'true';
        });
    },

    /**
     * 判断文本是否看起来像代码
     */
    looksLikeCode: function(text) {
        // 包含典型的代码特征
        const codePatterns = [
            /\bfloat[234]?\b/,
            /\bhalf[234]?\b/,
            /\bint[234]?\b/,
            /\bvoid\b/,
            /\bstruct\b/,
            /\breturn\b/,
            /\bif\b.*\(/,
            /\bfor\b.*\(/,
            /\bwhile\b.*\(/,
            /[=+\-*/]=?/,
            /\bSAMPLE_TEXTURE/,
            /\bTransform\w+To\w+/,
            /\bunity_\w+/,
            /\bUNITY_MATRIX/,
            /\bmul\s*\(/,
            /:\s*[A-Z]+\d*\s*[;{]?/,  // 语义如 : POSITION
            /#pragma\b/,
            /#include\b/,
        ];
        
        return codePatterns.some(pattern => pattern.test(text));
    }
};

// HTML 转义函数
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// 页面加载后自动初始化
document.addEventListener('DOMContentLoaded', function() {
    ShaderHighlight.initCodeBlocks();
});

// 导出供其他脚本使用
window.ShaderHighlight = ShaderHighlight;
