// 统一的导航和搜索功能

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    initNavigation();
    initSearch();
    highlightActiveLink();
});

// 初始化导航
function initNavigation() {
    const currentPath = window.location.pathname;
    const currentPage = currentPath.split('/').pop() || 'index.html';
    const links = document.querySelectorAll('.nav-section a');
    
    links.forEach(link => {
        const linkPage = link.getAttribute('href');
        // 支持相对路径和绝对路径
        const linkFileName = linkPage.split('/').pop();
        const currentFileName = currentPage.split('/').pop();
        
        if (linkFileName === currentFileName || 
            (currentPage === '' && linkPage === 'index.html') ||
            (currentPage === 'index.html' && linkPage === 'index.html') ||
            currentPath.includes(linkPage)) {
            link.classList.add('active');
        }
    });
}

// 初始化搜索功能
function initSearch() {
    const searchInput = document.getElementById('searchInput');
    const searchBtn = document.getElementById('searchBtn');
    
    if (!searchInput) return;
    
    // 搜索函数列表（支持 html/ 子目录）
    const functions = [
        // 入门指南
        { name: '教程介绍', title: '教程介绍', url: 'html/00-introduction.html', keywords: '介绍 introduction 入门' },
        { name: 'URP概述', title: 'URP 概述', url: 'html/01-urp-overview.html', keywords: 'urp universal render pipeline 概述' },
        { name: 'Shader基础', title: 'Shader 基础', url: 'html/02-shader-basics.html', keywords: 'shader 基础 basics' },
        { name: '术语黑话', title: '术语与黑话', url: 'html/03-terminology.html', keywords: '术语 黑话 terminology 暗号' },
        // 渲染流程
        { name: 'URP渲染管线', title: 'URP 渲染管线', url: 'html/urp-rendering-pipeline.html', keywords: 'urp 渲染 管线 pipeline rendering' },
        { name: '渲染阶段', title: '渲染阶段详解', url: 'html/rendering-stages.html', keywords: '渲染 阶段 stage' },
        // 坐标空间
        { name: '坐标系', title: '坐标系概述', url: 'html/coordinate-spaces.html', keywords: '坐标系 coordinate space' },
        { name: '模型空间', title: '模型空间详解', url: 'html/object-space.html', keywords: '模型 空间 object space' },
        { name: '世界空间', title: '世界空间详解', url: 'html/world-space.html', keywords: '世界 空间 world space' },
        { name: '观察空间', title: '观察空间详解', url: 'html/view-space.html', keywords: '观察 空间 view space camera' },
        { name: '裁剪空间', title: '裁剪空间详解', url: 'html/clip-space.html', keywords: '裁剪 空间 clip space' },
        { name: '屏幕空间', title: '屏幕空间详解', url: 'html/screen-space.html', keywords: '屏幕 空间 screen space' },
        { name: '切线空间', title: '切线空间详解', url: 'html/tangent-space.html', keywords: '切线 空间 tangent space' },
        { name: 'MVP矩阵', title: 'MVP 矩阵详解', url: 'html/mvp-matrix.html', keywords: 'mvp matrix 矩阵 model view projection' },
        // UV 变换
        { name: 'RotateUV', title: 'RotateUV - 旋转矩阵', url: 'html/rotate-uv.html', keywords: '旋转 矩阵 matrix rotation uv' },
        { name: 'UV滚动', title: 'UV 滚动', url: 'html/uv-scroll.html', keywords: '滚动 scroll uv 动画' },
        { name: 'UV缩放', title: 'UV 缩放', url: 'html/uv-scale.html', keywords: '缩放 scale uv' },
        { name: 'UV偏移', title: 'UV 偏移', url: 'html/uv-offset.html', keywords: '偏移 offset uv' },
        // 纹理操作
        { name: '纹理采样', title: '纹理采样', url: 'html/texture-sampling.html', keywords: '纹理 texture sampling 采样' },
        { name: '纹理寻址', title: '纹理寻址模式', url: 'html/texture-addressing.html', keywords: '纹理 寻址 addressing wrap clamp repeat' },
        { name: '纹理混合', title: '纹理混合', url: 'html/texture-blend.html', keywords: '纹理 混合 blend lerp' },
        // 光照计算
        { name: 'Lambert', title: 'Lambert 漫反射', url: 'html/lambert.html', keywords: 'lambert 漫反射 diffuse lighting' },
        { name: 'Phong', title: 'Phong 高光', url: 'html/phong.html', keywords: 'phong 高光 specular' },
        { name: 'BlinnPhong', title: 'Blinn-Phong', url: 'html/blinn-phong.html', keywords: 'blinn phong 高光' },
        // 特效函数
        { name: '溶解', title: '溶解效果', url: 'html/dissolve.html', keywords: '溶解 dissolve' },
        { name: '扭曲', title: '扭曲效果', url: 'html/distortion.html', keywords: '扭曲 distortion' },
        { name: '菲涅尔', title: '菲涅尔效果', url: 'html/fresnel.html', keywords: '菲涅尔 fresnel rim' }
    ];
    
    function performSearch() {
        const query = searchInput.value.toLowerCase().trim();
        if (!query) {
            clearSearchResults();
            return;
        }
        
        const results = functions.filter(func => {
            return func.name.toLowerCase().includes(query) ||
                   func.title.toLowerCase().includes(query) ||
                   func.keywords.toLowerCase().includes(query);
        });
        
        displaySearchResults(results, query);
    }
    
    if (searchBtn) {
        searchBtn.addEventListener('click', performSearch);
    }
    
    searchInput.addEventListener('input', performSearch);
    searchInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            performSearch();
        }
    });
}

// 显示搜索结果
function displaySearchResults(results, query) {
    // 移除旧的搜索结果
    clearSearchResults();
    
    if (results.length === 0) {
        const noResults = document.createElement('div');
        noResults.className = 'search-results';
        noResults.innerHTML = '<p style="padding: 20px; color: rgba(255,255,255,0.6);">未找到匹配的函数</p>';
        document.querySelector('.nav-menu').appendChild(noResults);
        return;
    }
    
    const resultsDiv = document.createElement('div');
    resultsDiv.className = 'search-results';
    resultsDiv.style.background = 'rgba(74, 144, 226, 0.1)';
    resultsDiv.style.margin = '10px';
    resultsDiv.style.padding = '10px';
    resultsDiv.style.borderRadius = '4px';
    resultsDiv.style.border = '1px solid rgba(74, 144, 226, 0.3)';
    
    const title = document.createElement('h4');
    title.textContent = `搜索结果 (${results.length})`;
    title.style.color = 'white';
    title.style.marginBottom = '10px';
    title.style.fontSize = '12px';
    resultsDiv.appendChild(title);
    
    const resultsList = document.createElement('ul');
    resultsList.style.listStyle = 'none';
    
    results.forEach(func => {
        const li = document.createElement('li');
        const a = document.createElement('a');
        a.href = func.url;
        a.textContent = func.title;
        a.style.display = 'block';
        a.style.padding = '8px 15px';
        a.style.color = 'white';
        a.style.textDecoration = 'none';
        a.style.borderRadius = '3px';
        a.style.transition = 'background 0.3s';
        
        a.addEventListener('mouseenter', () => {
            a.style.background = 'rgba(255,255,255,0.2)';
        });
        a.addEventListener('mouseleave', () => {
            a.style.background = 'transparent';
        });
        
        li.appendChild(a);
        resultsList.appendChild(li);
    });
    
    resultsDiv.appendChild(resultsList);
    document.querySelector('.nav-menu').appendChild(resultsDiv);
}

// 清除搜索结果
function clearSearchResults() {
    const oldResults = document.querySelector('.search-results');
    if (oldResults) {
        oldResults.remove();
    }
}

// 高亮当前活动的链接
function highlightActiveLink() {
    const currentPage = window.location.pathname.split('/').pop() || 'index.html';
    const links = document.querySelectorAll('.nav-section a');
    
    links.forEach(link => {
        link.classList.remove('active');
        const linkPage = link.getAttribute('href');
        if (linkPage === currentPage || (currentPage === '' && linkPage === 'index.html')) {
            link.classList.add('active');
        }
    });
}

// 工具函数：格式化代码
function formatCode(code) {
    // 简单的代码高亮（可以后续扩展）
    return code
        .replace(/\/\/.*$/gm, '<span class="comment">$&</span>')
        .replace(/\b(float|half|int|bool|return|if|else|for|while)\b/g, '<span class="keyword">$1</span>')
        .replace(/\b(\w+)\s*\(/g, '<span class="function">$1</span>(')
        .replace(/"([^"]*)"/g, '<span class="string">"$1"</span>')
        .replace(/\b(\d+\.?\d*)\b/g, '<span class="number">$1</span>');
}

// 导出供其他页面使用
window.shaderLibrary = {
    formatCode,
    highlightActiveLink
};
