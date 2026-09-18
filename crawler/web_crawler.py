
import requests
from bs4 import BeautifulSoup
import re
import time
from urllib.parse import urljoin, urlparse

class WebCrawler:
    """AI's eyes on the internet"""
    
    def __init__(self):
        self.visited = set()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MyAI/1.0 (Learning Bot)'
        })
    
    def fetch(self, url):
        """Get webpage content"""
        try:
            if url in self.visited:
                return None
            
            self.visited.add(url)
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            return response.text
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def extract_text(self, html):
        """Extract readable text from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(['script', 'style', 'nav', 'footer', 'header']):
            script.decompose()
        
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = ' '.join(lines)
        
        # Limit length
        return text[:5000]
    
    def extract_links(self, html, base_url):
        """Find links to crawl"""
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        
        for anchor in soup.find_all('a', href=True):
            href = anchor['href']
            full_url = urljoin(base_url, href)
            
            # Only same domain
            if urlparse(full_url).netloc == urlparse(base_url).netloc:
                links.append(full_url)
        
        return list(set(links))[:10]  # Limit links
    
    def search_and_learn(self, query, memory_system, max_pages=3):
        """Search topic and learn from results"""
        # Simple search via DuckDuckGo or direct URLs
        search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        
        html = self.fetch(search_url)
        if not html:
            return []
        
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        
        # Extract search results
        for link in soup.find_all('a', {'class': 'result__a'}):
            href = link.get('href')
            if href and href.startswith('http'):
                page_html = self.fetch(href)
                if page_html:
                    text = self.extract_text(page_html)
                    if len(text) > 100:
                        # Store in memory
                        memory_system.store_knowledge(
                            topic=query,
                            content=text[:2000],
                            source=href,
                            importance=len(text) / 5000
                        )
                        results.append({
                            'source': href,
                            'content': text[:500]
                        })
                        
                        if len(results) >= max_pages:
                            break
        
        return results
    
    def crawl_topic(self, start_url, memory_system, depth=2):
        """Deep crawl a website"""
        to_visit = [(start_url, 0)]
        learned = []
        
        while to_visit:
            url, current_depth = to_visit.pop(0)
            
            if current_depth > depth:
                continue
            
            html = self.fetch(url)
            if not html:
                continue
            
            text = self.extract_text(html)
            if text:
                memory_system.store_knowledge(
                    topic=urlparse(url).path,
                    content=text[:2000],
                    source=url,
                    importance=0.5
                )
                learned.append(url)
            
            if current_depth < depth:
                links = self.extract_links(html, url)
                for link in links:
                    if link not in self.visited:
                        to_visit.append((link, current_depth + 1))
            
            time.sleep(1)  # Be polite
        
        return learned
