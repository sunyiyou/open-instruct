// Main JavaScript for Evaluation Results Viewer

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 100);
    });

    // Enhanced table interactions
    const tableRows = document.querySelectorAll('.table-hover tbody tr');
    tableRows.forEach(row => {
        row.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.02)';
            this.style.zIndex = '10';
            this.style.position = 'relative';
            this.style.boxShadow = '0 4px 20px rgba(0,0,0,0.1)';
        });
        
        row.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1)';
            this.style.zIndex = 'auto';
            this.style.position = 'static';
            this.style.boxShadow = 'none';
        });
    });

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Copy to clipboard functionality
    function addCopyButtons() {
        const codeBlocks = document.querySelectorAll('.code-block');
        codeBlocks.forEach(block => {
            const copyButton = document.createElement('button');
            copyButton.className = 'btn btn-sm btn-outline-secondary copy-btn';
            copyButton.innerHTML = '<i class="fas fa-copy"></i>';
            copyButton.style.position = 'absolute';
            copyButton.style.top = '10px';
            copyButton.style.right = '10px';
            copyButton.style.zIndex = '1000';
            
            const container = block.parentNode;
            container.style.position = 'relative';
            container.appendChild(copyButton);
            
            copyButton.addEventListener('click', async () => {
                try {
                    await navigator.clipboard.writeText(block.textContent);
                    copyButton.innerHTML = '<i class="fas fa-check"></i>';
                    copyButton.classList.remove('btn-outline-secondary');
                    copyButton.classList.add('btn-success');
                    
                    setTimeout(() => {
                        copyButton.innerHTML = '<i class="fas fa-copy"></i>';
                        copyButton.classList.remove('btn-success');
                        copyButton.classList.add('btn-outline-secondary');
                    }, 2000);
                } catch (err) {
                    console.error('Failed to copy text: ', err);
                }
            });
        });
    }
    
    addCopyButtons();

    // Search functionality for tables
    function addTableSearch() {
        const tables = document.querySelectorAll('.table');
        tables.forEach(table => {
            const searchContainer = document.createElement('div');
            searchContainer.className = 'mb-3';
            searchContainer.innerHTML = `
                <div class="input-group">
                    <span class="input-group-text">
                        <i class="fas fa-search"></i>
                    </span>
                    <input type="text" class="form-control table-search" placeholder="Search table...">
                </div>
            `;
            
            table.parentNode.insertBefore(searchContainer, table);
            
            const searchInput = searchContainer.querySelector('.table-search');
            searchInput.addEventListener('input', function() {
                const filter = this.value.toLowerCase();
                const rows = table.querySelectorAll('tbody tr');
                
                rows.forEach(row => {
                    const text = row.textContent.toLowerCase();
                    row.style.display = text.includes(filter) ? '' : 'none';
                });
            });
        });
    }

    // Add table search if there are tables with more than 5 rows
    const largeTables = document.querySelectorAll('.table tbody tr');
    if (largeTables.length > 5) {
        addTableSearch();
    }

    // Enhanced metric cards with progress indicators
    function animateMetrics() {
        const metricValues = document.querySelectorAll('.metric-value');
        metricValues.forEach(metric => {
            const value = parseFloat(metric.textContent);
            if (!isNaN(value) && value <= 1) {
                // Animate percentage values
                let current = 0;
                const increment = value / 50;
                const timer = setInterval(() => {
                    current += increment;
                    if (current >= value) {
                        current = value;
                        clearInterval(timer);
                    }
                    metric.textContent = current.toFixed(3);
                }, 20);
            }
        });
    }

    // Only animate on first load
    if (!sessionStorage.getItem('animated')) {
        animateMetrics();
        sessionStorage.setItem('animated', 'true');
    }

    // Dynamic loading indicators
    function showLoading(element) {
        element.classList.add('loading');
    }

    function hideLoading(element) {
        element.classList.remove('loading');
    }

    // Export functionality
    function addExportButtons() {
        const tables = document.querySelectorAll('.table');
        tables.forEach(table => {
            const exportBtn = document.createElement('button');
            exportBtn.className = 'btn btn-outline-primary btn-sm me-2';
            exportBtn.innerHTML = '<i class="fas fa-download me-1"></i>Export CSV';
            
            const tableContainer = table.closest('.card-body');
            if (tableContainer) {
                const header = tableContainer.previousElementSibling;
                if (header && header.classList.contains('card-header')) {
                    header.appendChild(exportBtn);
                    
                    exportBtn.addEventListener('click', () => {
                        exportTableToCSV(table);
                    });
                }
            }
        });
    }

    function exportTableToCSV(table) {
        const rows = table.querySelectorAll('tr');
        const csvContent = [];
        
        rows.forEach(row => {
            const cols = row.querySelectorAll('td, th');
            const rowData = [];
            cols.forEach(col => {
                // Clean up the text content
                let text = col.textContent.trim();
                text = text.replace(/[\n\r]+/g, ' ');
                text = text.replace(/,/g, ';');
                rowData.push(text);
            });
            csvContent.push(rowData.join(','));
        });
        
        const csvString = csvContent.join('\n');
        const blob = new Blob([csvString], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = 'evaluation_results.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    }

    addExportButtons();

    // Dark mode toggle (optional)
    function addDarkModeToggle() {
        const navbar = document.querySelector('.navbar');
        if (navbar) {
            const darkModeBtn = document.createElement('button');
            darkModeBtn.className = 'btn btn-outline-light btn-sm';
            darkModeBtn.innerHTML = '<i class="fas fa-moon"></i>';
            darkModeBtn.id = 'darkModeToggle';
            
            const navbarNav = navbar.querySelector('.navbar-nav');
            if (navbarNav) {
                const li = document.createElement('li');
                li.className = 'nav-item';
                li.appendChild(darkModeBtn);
                navbarNav.appendChild(li);
                
                // Check for saved dark mode preference
                if (localStorage.getItem('darkMode') === 'true') {
                    enableDarkMode();
                }
                
                darkModeBtn.addEventListener('click', toggleDarkMode);
            }
        }
    }

    function toggleDarkMode() {
        if (document.body.classList.contains('dark-mode')) {
            disableDarkMode();
        } else {
            enableDarkMode();
        }
    }

    function enableDarkMode() {
        document.body.classList.add('dark-mode');
        document.querySelector('#darkModeToggle').innerHTML = '<i class="fas fa-sun"></i>';
        localStorage.setItem('darkMode', 'true');
        
        // Add dark mode styles
        const darkModeCSS = `
            .dark-mode {
                background-color: #1a1a1a !important;
                color: #ffffff !important;
            }
            .dark-mode .card {
                background-color: #2d2d2d !important;
                border-color: #404040 !important;
            }
            .dark-mode .table {
                background-color: #2d2d2d !important;
                color: #ffffff !important;
            }
            .dark-mode .code-block {
                background-color: #1e1e1e !important;
                border-color: #404040 !important;
                color: #ffffff !important;
            }
        `;
        
        let styleSheet = document.getElementById('darkModeStyles');
        if (!styleSheet) {
            styleSheet = document.createElement('style');
            styleSheet.id = 'darkModeStyles';
            document.head.appendChild(styleSheet);
        }
        styleSheet.textContent = darkModeCSS;
    }

    function disableDarkMode() {
        document.body.classList.remove('dark-mode');
        document.querySelector('#darkModeToggle').innerHTML = '<i class="fas fa-moon"></i>';
        localStorage.setItem('darkMode', 'false');
        
        const styleSheet = document.getElementById('darkModeStyles');
        if (styleSheet) {
            styleSheet.remove();
        }
    }

    // addDarkModeToggle(); // Uncomment to enable dark mode

    // Performance monitoring
    window.addEventListener('load', () => {
        const loadTime = performance.now();
        console.log(`Page loaded in ${loadTime.toFixed(2)}ms`);
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + K for search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.querySelector('.table-search');
            if (searchInput) {
                searchInput.focus();
            }
        }
        
        // Escape to clear search
        if (e.key === 'Escape') {
            const searchInputs = document.querySelectorAll('.table-search');
            searchInputs.forEach(input => {
                input.value = '';
                input.dispatchEvent(new Event('input'));
            });
        }
    });

    // Add keyboard shortcut hints
    const helpBtn = document.createElement('button');
    helpBtn.className = 'btn btn-outline-secondary btn-sm position-fixed';
    helpBtn.style.bottom = '20px';
    helpBtn.style.right = '20px';
    helpBtn.style.zIndex = '1000';
    helpBtn.innerHTML = '<i class="fas fa-question"></i>';
    helpBtn.title = 'Keyboard shortcuts: Ctrl+K (search), Esc (clear)';
    helpBtn.setAttribute('data-bs-toggle', 'tooltip');
    helpBtn.setAttribute('data-bs-placement', 'left');
    
    document.body.appendChild(helpBtn);
    
    // Initialize tooltip for help button
    new bootstrap.Tooltip(helpBtn);
});

// Utility functions for external use
window.evalViewer = {
    showLoading: function(selector) {
        const element = document.querySelector(selector);
        if (element) element.classList.add('loading');
    },
    
    hideLoading: function(selector) {
        const element = document.querySelector(selector);
        if (element) element.classList.remove('loading');
    },
    
    refreshCharts: function() {
        // Trigger chart refresh if needed
        window.dispatchEvent(new Event('resize'));
    },
    
    scrollToTop: function() {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
};
