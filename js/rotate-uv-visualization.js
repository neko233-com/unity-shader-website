// RotateUV 可视化脚本

document.addEventListener('DOMContentLoaded', function() {
    const canvas = document.getElementById('rotationCanvas');
    if (!canvas) return; // 如果页面没有这个 canvas，直接返回
    
    const ctx = canvas.getContext('2d');
    
    // 控制元素
    const angleSlider = document.getElementById('angleSlider');
    const centerXSlider = document.getElementById('centerXSlider');
    const centerYSlider = document.getElementById('centerYSlider');
    const angleValue = document.getElementById('angleValue');
    const centerXValue = document.getElementById('centerXValue');
    const centerYValue = document.getElementById('centerYValue');
    const showGrid = document.getElementById('showGrid');
    const showOriginal = document.getElementById('showOriginal');
    const animateRotation = document.getElementById('animateRotation');
    
    // 状态
    let angle = 0;
    let centerX = 0.5;
    let centerY = 0.5;
    let animationAngle = 0;
    
    // 设置画布尺寸
    function resizeCanvas() {
        const container = canvas.parentElement;
        const maxWidth = Math.min(800, container.clientWidth - 40);
        canvas.width = maxWidth;
        canvas.height = (maxWidth * 3) / 4; // 4:3 比例
    }
    
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    // 将 UV 坐标转换为画布坐标
    function uvToCanvas(u, v) {
        const padding = 50;
        const w = canvas.width - padding * 2;
        const h = canvas.height - padding * 2;
        return {
            x: padding + u * w,
            y: padding + (1 - v) * h // 翻转 Y 轴
        };
    }
    
    // 旋转 UV 坐标
    function rotateUV(uv, rotation, center) {
        // 1. 移到中心
        let x = uv.x - center.x;
        let y = uv.y - center.y;
        
        // 2. 计算 sin 和 cos
        const s = Math.sin(rotation);
        const c = Math.cos(rotation);
        
        // 3. 应用旋转矩阵
        const xNew = x * c - y * s;
        const yNew = x * s + y * c;
        
        // 4. 移回
        return {
            x: xNew + center.x,
            y: yNew + center.y
        };
    }
    
    // 绘制网格
    function drawGrid() {
        if (!showGrid || !showGrid.checked) return;
        
        ctx.strokeStyle = '#e0e0e0';
        ctx.lineWidth = 1;
        
        const padding = 50;
        const w = canvas.width - padding * 2;
        const h = canvas.height - padding * 2;
        
        // 绘制 UV 网格 (0-1 范围)
        for (let i = 0; i <= 10; i++) {
            const u = i / 10;
            const pos = uvToCanvas(u, 0);
            
            // 垂直线
            ctx.beginPath();
            ctx.moveTo(pos.x, padding);
            ctx.lineTo(pos.x, padding + h);
            ctx.stroke();
            
            // 水平线
            const v = i / 10;
            const pos2 = uvToCanvas(0, v);
            ctx.beginPath();
            ctx.moveTo(padding, pos2.y);
            ctx.lineTo(padding + w, pos2.y);
            ctx.stroke();
        }
        
        // 绘制中心线
        ctx.strokeStyle = '#4a90e2';
        ctx.lineWidth = 2;
        const centerPos = uvToCanvas(0.5, 0.5);
        ctx.beginPath();
        ctx.moveTo(centerPos.x, padding);
        ctx.lineTo(centerPos.x, padding + h);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(padding, centerPos.y);
        ctx.lineTo(padding + w, centerPos.y);
        ctx.stroke();
    }
    
    // 绘制测试点
    function drawPoints() {
        const testPoints = [
            { uv: { x: 0.2, y: 0.2 }, color: '#ff6b6b', label: 'A' },
            { uv: { x: 0.8, y: 0.2 }, color: '#4ecdc4', label: 'B' },
            { uv: { x: 0.8, y: 0.8 }, color: '#45b7d1', label: 'C' },
            { uv: { x: 0.2, y: 0.8 }, color: '#f9ca24', label: 'D' },
            { uv: { x: 0.5, y: 0.5 }, color: '#6c5ce7', label: 'O' }
        ];
        
        const center = { x: centerX, y: centerY };
        const rotation = angle * Math.PI / 180;
        
        testPoints.forEach(point => {
            // 原始位置
            if (showOriginal && showOriginal.checked) {
                const origPos = uvToCanvas(point.uv.x, point.uv.y);
                ctx.fillStyle = point.color;
                ctx.globalAlpha = 0.3;
                ctx.beginPath();
                ctx.arc(origPos.x, origPos.y, 8, 0, Math.PI * 2);
                ctx.fill();
                ctx.globalAlpha = 1.0;
            }
            
            // 旋转后的位置
            const rotated = rotateUV(point.uv, rotation, center);
            const newPos = uvToCanvas(rotated.x, rotated.y);
            
            ctx.fillStyle = point.color;
            ctx.beginPath();
            ctx.arc(newPos.x, newPos.y, 10, 0, Math.PI * 2);
            ctx.fill();
            
            // 标签
            ctx.fillStyle = '#333';
            ctx.font = 'bold 14px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(point.label, newPos.x, newPos.y - 15);
            
            // 连线（如果显示原始位置）
            if (showOriginal && showOriginal.checked) {
                const origPos = uvToCanvas(point.uv.x, point.uv.y);
                ctx.strokeStyle = point.color;
                ctx.lineWidth = 2;
                ctx.globalAlpha = 0.3;
                ctx.setLineDash([5, 5]);
                ctx.beginPath();
                ctx.moveTo(origPos.x, origPos.y);
                ctx.lineTo(newPos.x, newPos.y);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.globalAlpha = 1.0;
            }
        });
        
        // 绘制旋转中心
        const centerPos = uvToCanvas(center.x, center.y);
        ctx.fillStyle = '#e74c3c';
        ctx.beginPath();
        ctx.arc(centerPos.x, centerPos.y, 6, 0, Math.PI * 2);
        ctx.fill();
        
        // 绘制旋转箭头
        ctx.strokeStyle = '#e74c3c';
        ctx.lineWidth = 3;
        const arrowLength = 40;
        const arrowAngle = rotation;
        const arrowEndX = centerPos.x + Math.cos(arrowAngle) * arrowLength;
        const arrowEndY = centerPos.y - Math.sin(arrowAngle) * arrowLength;
        
        ctx.beginPath();
        ctx.moveTo(centerPos.x, centerPos.y);
        ctx.lineTo(arrowEndX, arrowEndY);
        ctx.stroke();
        
        // 绘制角度文本
        ctx.fillStyle = '#e74c3c';
        ctx.font = 'bold 16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`${angle.toFixed(1)}°`, centerPos.x, centerPos.y - 30);
    }
    
    // 绘制 UV 坐标轴标签
    function drawLabels() {
        const padding = 50;
        ctx.fillStyle = '#666';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        
        // U 轴标签
        for (let i = 0; i <= 1; i += 0.2) {
            const pos = uvToCanvas(i, 0);
            ctx.fillText(i.toFixed(1), pos.x, canvas.height - 10);
        }
        ctx.fillText('U', canvas.width / 2, canvas.height - 10);
        
        // V 轴标签
        ctx.save();
        ctx.translate(20, canvas.height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('V', 0, 0);
        ctx.restore();
        
        for (let i = 0; i <= 1; i += 0.2) {
            const pos = uvToCanvas(0, i);
            ctx.fillText(i.toFixed(1), 20, pos.y);
        }
    }
    
    // 主绘制函数
    function draw() {
        // 清空画布
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // 绘制背景
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // 绘制网格
        drawGrid();
        
        // 绘制点
        drawPoints();
        
        // 绘制标签
        drawLabels();
        
        // 绘制标题
        ctx.fillStyle = '#333';
        ctx.font = 'bold 18px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('UV 旋转可视化', canvas.width / 2, 25);
    }
    
    // 事件监听
    if (angleSlider) {
        angleSlider.addEventListener('input', function() {
            angle = parseFloat(this.value);
            if (angleValue) angleValue.textContent = angle.toFixed(1);
            draw();
        });
    }
    
    if (centerXSlider) {
        centerXSlider.addEventListener('input', function() {
            centerX = parseFloat(this.value);
            if (centerXValue) centerXValue.textContent = centerX.toFixed(2);
            draw();
        });
    }
    
    if (centerYSlider) {
        centerYSlider.addEventListener('input', function() {
            centerY = parseFloat(this.value);
            if (centerYValue) centerYValue.textContent = centerY.toFixed(2);
            draw();
        });
    }
    
    if (showGrid) showGrid.addEventListener('change', draw);
    if (showOriginal) showOriginal.addEventListener('change', draw);
    
    // 自动旋转动画
    let animationId = null;
    if (animateRotation) {
        animateRotation.addEventListener('change', function() {
            if (this.checked) {
                function animate() {
                    animationAngle += 2; // 每帧增加 2 度
                    if (animationAngle >= 360) animationAngle = 0;
                    if (angleSlider) angleSlider.value = animationAngle;
                    angle = animationAngle;
                    if (angleValue) angleValue.textContent = angle.toFixed(1);
                    draw();
                    animationId = requestAnimationFrame(animate);
                }
                animate();
            } else {
                if (animationId) {
                    cancelAnimationFrame(animationId);
                }
            }
        });
    }
    
    // 初始绘制
    draw();
});
