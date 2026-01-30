/**
 * AI学习平台 - 主应用 v2.0
 * 左侧Tab垂直导航布局
 */

class AILearningApp {
    constructor() {
        this.data = null;
        this.currentCategory = null;
        this.currentChapter = null;
        this.currentSection = 'terminology';
    }

    async init() {
        try {
            await this.loadData();
            this.render();
            this.bindEvents();
        } catch (error) {
            console.error('初始化失败:', error);
            document.getElementById('content-body').innerHTML = `
                <div style="text-align:center;padding:50px;color:#e53e3e;">
                    <h3>加载失败</h3>
                    <p>${error.message}</p>
                    <button onclick="location.reload()" style="margin-top:20px;padding:10px 20px;cursor:pointer;">
                        刷新重试
                    </button>
                </div>
            `;
        }
    }

    async loadData() {
        const response = await fetch('./src/data/knowledge-base.json');
        if (!response.ok) throw new Error('无法加载知识库');
        this.data = await response.json();

        // 设置默认选中
        if (this.data.categories.length > 0) {
            this.currentCategory = this.data.categories[0];
            if (this.currentCategory.chapters.length > 0) {
                this.currentChapter = this.currentCategory.chapters[0];
            }
        }
    }

    render() {
        this.renderSidebar();
        this.renderContent();
    }

    renderSidebar() {
        const sidebar = document.getElementById('sidebar-nav');
        if (!sidebar) return;

        let html = '';

        this.data.categories.forEach(category => {
            html += `
                <div class="nav-category">
                    <div class="category-title">
                        <span class="icon">${category.icon}</span>
                        <span>${category.name}</span>
                    </div>
                    <ul class="chapter-list">
            `;

            category.chapters.forEach(chapter => {
                const isActive = this.currentChapter && this.currentChapter.id === chapter.id;
                html += `
                    <li class="chapter-item">
                        <div class="chapter-link ${isActive ? 'active' : ''}"
                             data-category="${category.id}"
                             data-chapter="${chapter.id}">
                            <span class="icon">${chapter.icon}</span>
                            <span class="title">${chapter.title}</span>
                        </div>
                    </li>
                `;
            });

            html += `</ul></div>`;
        });

        sidebar.innerHTML = html;
    }

    renderContent() {
        if (!this.currentChapter) return;

        // 更新面包屑
        const breadcrumb = document.getElementById('breadcrumb');
        if (breadcrumb) {
            breadcrumb.innerHTML = `
                <span>${this.currentCategory.name}</span>
                <span class="separator">/</span>
                <span class="current">${this.currentChapter.title}</span>
            `;
        }

        // 更新标题
        const titleEl = document.getElementById('chapter-title');
        if (titleEl) {
            titleEl.innerHTML = `
                <span class="icon">${this.currentChapter.icon}</span>
                <h2>${this.currentChapter.title}</h2>
            `;
        }

        // 更新Section标签
        this.renderSectionTabs();

        // 更新内容
        this.renderSectionContent();
    }

    renderSectionTabs() {
        const tabs = document.getElementById('section-tabs');
        if (!tabs || !this.currentChapter) return;

        const sections = this.currentChapter.sections;
        const sectionOrder = ['terminology', 'basic', 'advanced', 'practice'];

        let html = '';
        sectionOrder.forEach(key => {
            if (sections[key]) {
                const isActive = this.currentSection === key;
                html += `
                    <button class="section-tab ${isActive ? 'active' : ''}" data-section="${key}">
                        ${sections[key].title}
                    </button>
                `;
            }
        });

        tabs.innerHTML = html;
    }

    renderSectionContent() {
        const contentBody = document.getElementById('content-body');
        if (!contentBody || !this.currentChapter) return;

        const section = this.currentChapter.sections[this.currentSection];
        if (!section) return;

        let html = '';

        if (this.currentSection === 'terminology') {
            // 术语表格
            html = `
                <div class="terminology-table">
                    <table>
                        <thead>
                            <tr>
                                <th>术语</th>
                                <th>英文</th>
                                <th>说明</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${section.items.map(item => `
                                <tr>
                                    <td class="term-name">${item.term}</td>
                                    <td class="term-english">${item.english}</td>
                                    <td class="term-desc">${item.desc}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            `;
        } else {
            // Markdown内容
            html = `
                <div class="markdown-content">
                    ${this.renderMarkdown(section.content)}
                </div>
            `;
        }

        contentBody.innerHTML = html;

        // 代码高亮
        this.highlightCode();
    }

    highlightCode() {
        // 使用 highlight.js 高亮代码块
        if (typeof hljs !== 'undefined') {
            document.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightElement(block);
            });
        }
    }

    renderMarkdown(content) {
        // 简单的Markdown渲染
        let html = content;

        // 代码块
        html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
            return `<pre><code class="language-${lang}">${this.escapeHtml(code.trim())}</code></pre>`;
        });

        // 表格
        html = html.replace(/\|(.+)\|\n\|[-| ]+\|\n((?:\|.+\|\n?)+)/g, (match, header, body) => {
            const headers = header.split('|').filter(h => h.trim());
            const rows = body.trim().split('\n').map(row =>
                row.split('|').filter(c => c.trim())
            );

            return `
                <table>
                    <thead>
                        <tr>${headers.map(h => `<th>${h.trim()}</th>`).join('')}</tr>
                    </thead>
                    <tbody>
                        ${rows.map(row => `<tr>${row.map(c => `<td>${c.trim()}</td>`).join('')}</tr>`).join('')}
                    </tbody>
                </table>
            `;
        });

        // 标题
        html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');

        // 粗体
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

        // 行内代码
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

        // 列表
        html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');

        // 数字列表
        html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

        // 段落
        html = html.replace(/\n\n/g, '</p><p>');
        html = '<p>' + html + '</p>';
        html = html.replace(/<p><(h[23]|ul|ol|table|pre)/g, '<$1');
        html = html.replace(/<\/(h[23]|ul|ol|table|pre)><\/p>/g, '</$1>');
        html = html.replace(/<p><\/p>/g, '');

        return html;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    bindEvents() {
        // 章节点击
        document.getElementById('sidebar-nav')?.addEventListener('click', (e) => {
            const link = e.target.closest('.chapter-link');
            if (!link) return;

            const categoryId = link.dataset.category;
            const chapterId = link.dataset.chapter;

            this.currentCategory = this.data.categories.find(c => c.id === categoryId);
            this.currentChapter = this.currentCategory?.chapters.find(ch => ch.id === chapterId);
            this.currentSection = 'terminology';

            this.render();

            // 移动端关闭侧边栏
            document.getElementById('sidebar')?.classList.remove('open');
        });

        // Section Tab点击
        document.getElementById('section-tabs')?.addEventListener('click', (e) => {
            const tab = e.target.closest('.section-tab');
            if (!tab) return;

            this.currentSection = tab.dataset.section;
            this.renderSectionTabs();
            this.renderSectionContent();
        });

        // 移动端侧边栏切换
        document.getElementById('sidebar-toggle')?.addEventListener('click', () => {
            document.getElementById('sidebar')?.classList.toggle('open');
        });

        // 点击内容区关闭侧边栏
        document.getElementById('main-content')?.addEventListener('click', () => {
            document.getElementById('sidebar')?.classList.remove('open');
        });
    }
}

// 启动应用
document.addEventListener('DOMContentLoaded', () => {
    const app = new AILearningApp();
    app.init();
});
