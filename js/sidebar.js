/**
 * 统一的侧边栏组件
 * 支持折叠/展开，所有页面共享同一个侧边栏结构
 */

// 侧边栏导航数据
const sidebarData = [
    {
        id: 'getting-started',
        title: '🚀 入门指南',
        collapsed: false,
        items: [
            { title: '教程介绍', url: '00-introduction.html' },
            { title: 'URP 概述', url: '01-urp-overview.html' },
            { title: 'Shader 基础', url: '02-shader-basics.html' },
            { title: '术语与黑话', url: '03-terminology.html' }
        ]
    },
    {
        id: 'rendering-pipeline',
        title: '🔄 渲染流程',
        collapsed: true,
        items: [
            { title: 'URP 渲染管线', url: 'urp-rendering-pipeline.html' },
            { title: '渲染阶段详解', url: 'rendering-stages.html' },
            { title: 'Pass 类型', url: 'pass-types.html' },
            { title: 'Render Features', url: 'render-features.html' }
        ]
    },
    {
        id: 'coordinate-spaces',
        title: '📐 坐标空间',
        collapsed: true,
        items: [
            { title: '坐标系概述', url: 'coordinate-spaces.html' },
            { title: '模型空间详解', url: 'object-space.html' },
            { title: '世界空间详解', url: 'world-space.html' },
            { title: '观察空间详解', url: 'view-space.html' },
            { title: '裁剪空间详解', url: 'clip-space.html' },
            { title: '屏幕空间详解', url: 'screen-space.html' },
            { title: '切线空间详解', url: 'tangent-space.html' },
            { title: 'MVP 矩阵详解', url: 'mvp-matrix.html' }
        ]
    },
    {
        id: 'uv-transform',
        title: '🎨 UV 变换',
        collapsed: true,
        items: [
            { title: 'RotateUV - 旋转矩阵', url: 'rotate-uv.html' },
            { title: 'UV 滚动', url: 'uv-scroll.html' },
            { title: 'UV 缩放', url: 'uv-scale.html' },
            { title: 'UV 偏移', url: 'uv-offset.html' },
            { title: 'UV 动画', url: 'uv-animation.html' }
        ]
    },
    {
        id: 'texture-ops',
        title: '🖼️ 纹理操作',
        collapsed: true,
        items: [
            { title: '纹理采样', url: 'texture-sampling.html' },
            { title: '纹理寻址模式', url: 'texture-addressing.html' },
            { title: '纹理混合', url: 'texture-blend.html' },
            { title: 'Mipmap 详解', url: 'mipmap.html' }
        ]
    },
    {
        id: 'lighting',
        title: '💡 光照系统',
        collapsed: true,
        items: [
            { title: '光照概述', url: 'lighting-overview.html' },
            { title: 'Lambert 漫反射', url: 'lambert.html' },
            { title: 'Phong 高光', url: 'phong.html' },
            { title: 'Blinn-Phong', url: 'blinn-phong.html' },
            { title: 'PBR 光照', url: 'pbr-lighting.html' }
        ]
    },
    {
        id: 'effects',
        title: '✨ 特效函数',
        collapsed: true,
        items: [
            { title: '溶解效果', url: 'dissolve.html' },
            { title: '扭曲效果', url: 'distortion.html' },
            { title: '菲涅尔效果', url: 'fresnel.html' },
            { title: '边缘光', url: 'rim-light.html' }
        ]
    },
    {
        id: 'advanced',
        title: '🔧 高级主题',
        collapsed: true,
        items: [
            { title: 'Shader Variants', url: 'shader-variants.html' },
            { title: 'Compute Shader', url: 'compute-shader.html' },
            { title: '自定义 Render Pass', url: 'custom-render-pass.html' },
            { title: '性能优化', url: 'performance-optimization.html' }
        ]
    }
];

/**
 * 初始化侧边栏
 * @param {boolean} isSubPage - 是否是子页面（在 html/ 目录下）
 */
function initSidebar(isSubPage = false) {
    const navMenu = document.getElementById('navMenu');
    if (!navMenu) return;

    // 获取当前页面的文件名
    const currentPath = window.location.pathname;
    const currentPage = currentPath.split('/').pop() || 'index.html';

    // 构建基础URL（处理子页面路径）
    const baseUrl = isSubPage ? '' : 'html/';

    // 从 localStorage 读取折叠状态
    const collapsedState = JSON.parse(localStorage.getItem('sidebarCollapsedState') || '{}');

    // 清空现有内容
    navMenu.innerHTML = '';

    // 生成侧边栏 HTML
    sidebarData.forEach(section => {
        const sectionDiv = document.createElement('div');
        sectionDiv.className = 'nav-section';
        sectionDiv.dataset.sectionId = section.id;

        // 检查是否有当前页面在这个分组中
        const hasActivePage = section.items.some(item => {
            const itemPage = item.url.split('/').pop();
            return currentPage === itemPage;
        });

        // 如果有当前页面，自动展开
        const isCollapsed = hasActivePage ? false : (collapsedState[section.id] !== undefined ? collapsedState[section.id] : section.collapsed);

        // 创建分组标题（可点击折叠）
        const header = document.createElement('h3');
        header.className = 'nav-section-header' + (isCollapsed ? ' collapsed' : '');
        header.innerHTML = `
            <span class="collapse-icon">${isCollapsed ? '▶' : '▼'}</span>
            <span class="section-title">${section.title}</span>
        `;
        header.addEventListener('click', () => toggleSection(section.id, header));

        // 创建链接列表
        const ul = document.createElement('ul');
        ul.className = 'nav-section-list' + (isCollapsed ? ' collapsed' : '');

        section.items.forEach(item => {
            const li = document.createElement('li');
            const a = document.createElement('a');
            a.href = baseUrl + item.url;
            a.textContent = item.title;

            // 检查是否是当前页面
            const itemPage = item.url.split('/').pop();
            if (currentPage === itemPage) {
                a.classList.add('active');
            }

            li.appendChild(a);
            ul.appendChild(li);
        });

        sectionDiv.appendChild(header);
        sectionDiv.appendChild(ul);
        navMenu.appendChild(sectionDiv);
    });
}

/**
 * 切换分组的折叠状态
 */
function toggleSection(sectionId, headerElement) {
    const section = headerElement.parentElement;
    const list = section.querySelector('.nav-section-list');
    const icon = headerElement.querySelector('.collapse-icon');

    const isCollapsed = list.classList.contains('collapsed');

    if (isCollapsed) {
        // 展开
        list.classList.remove('collapsed');
        headerElement.classList.remove('collapsed');
        icon.textContent = '▼';
    } else {
        // 折叠
        list.classList.add('collapsed');
        headerElement.classList.add('collapsed');
        icon.textContent = '▶';
    }

    // 保存状态到 localStorage
    const collapsedState = JSON.parse(localStorage.getItem('sidebarCollapsedState') || '{}');
    collapsedState[sectionId] = !isCollapsed;
    localStorage.setItem('sidebarCollapsedState', JSON.stringify(collapsedState));
}

/**
 * 展开所有分组
 */
function expandAllSections() {
    document.querySelectorAll('.nav-section-list.collapsed').forEach(list => {
        list.classList.remove('collapsed');
    });
    document.querySelectorAll('.nav-section-header.collapsed').forEach(header => {
        header.classList.remove('collapsed');
        header.querySelector('.collapse-icon').textContent = '▼';
    });
    localStorage.setItem('sidebarCollapsedState', '{}');
}

/**
 * 折叠所有分组
 */
function collapseAllSections() {
    const collapsedState = {};
    document.querySelectorAll('.nav-section').forEach(section => {
        const sectionId = section.dataset.sectionId;
        const list = section.querySelector('.nav-section-list');
        const header = section.querySelector('.nav-section-header');
        
        // 如果没有 active 链接，才折叠
        if (!list.querySelector('a.active')) {
            list.classList.add('collapsed');
            header.classList.add('collapsed');
            header.querySelector('.collapse-icon').textContent = '▶';
            collapsedState[sectionId] = true;
        }
    });
    localStorage.setItem('sidebarCollapsedState', JSON.stringify(collapsedState));
}

// 导出函数供其他脚本使用
window.sidebarModule = {
    initSidebar,
    expandAllSections,
    collapseAllSections
};
