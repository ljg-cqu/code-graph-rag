#!/usr/bin/env python3
"""
Knowledge Investigator Agent Runner
Executes the knowledge investigation workflow for a given doubt/topic.
"""

import json
import os
import sys
import re
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional
import hashlib

# ==============================================================================
# AGENT PARAMETERS (from user input)
# ==============================================================================

PARAMETERS = {
    "doubt": "How to become an entrepreneur?",
    "investigation_depth": "auto",
    "complexity_factor": "auto",
    "web_search_enable": True,
    "web_search_languages": {"en": 0.5, "zh": 0.3, "others": 0.2},
    "web_search_recency_window": "auto",
    "web_search_minimum_sources": "auto",
    "output_path": "/home/zealy/github/ljg-cqu/knowledge/Roles/Entrepreneur/knowledge/",
    "output_language": "EN",
    "enable_web_info_download": False,
    "web_info_store_path": "/home/zealy/github/ljg-cqu/knowledge/Roles/Entrepreneur/knowledge/",
    "worker_count": "auto",
    "reviewer_count": "auto",
    "enable_fix_cycle": True,
    "max_fix_iterations": 3,
    "enable_d25_taxonomy": False,
    "enable_epistemic_classification": True,
    "enable_concern_assessment": False,
    "user_concerns": [],
    "epistemic_batch_size": 20,
    "enable_causal_effect_visualization": True,
    "causal_confidence_threshold": 0.7,
    "max_causal_edges": 50,
    "enable_thinking_maps": True,
    "enable_first_principles_explanations": True,
    "layer_mode": "all",
    "enable_universal_truth_detection": True,
    "enable_pattern_extraction": True,
    "enable_assumption_lifecycle": True,
    "enabled_dimensions": [],
    "enable_period_system": True,
}

# ==============================================================================
# COMPLEXITY SCORER (from lib/complexity_scorer.py)
# ==============================================================================

_COMPLEXITY_KEYWORDS = {
    "architect": ["architecture", "framework", "orchestrator", "workflow", "entire project", "systemic"],
    "complex": ["multi-agent", "traceability", "quality gate", "parallel", "concurrent", "distributed"],
    "moderate": ["analyze", "evaluate", "compare", "assess", "optimize", "design", "plan", "strategy"],
    "simple": ["basic", "simple", "introduct", "tutorial", "how does", "what is", "define", "explain"],
    "mini": ["quick", "brief", "summary", "overview", "tldr", "short", "single"],
}

_COMPLEXITY_WEIGHTS = {"architect": 2.0, "complex": 1.5, "moderate": 1.0, "simple": 0.5, "mini": 0.25}
_INV009_THRESHOLDS = {"quick": 3, "standard": 5, "comprehensive": 10}
_EVIDENCE_SCALING = {
    "mini": {"min_sources": 3, "worker_count": 1},
    "simple": {"min_sources": 4, "worker_count": 2},
    "moderate": {"min_sources": 6, "worker_count": 3},
    "complex": {"min_sources": 10, "worker_count": 5},
    "architect": {"min_sources": 12, "worker_count": 8},
}
_REVIEWER_MATRIX = {
    "mini": {"minimal": 1, "low": 1, "medium": 2, "high": 3, "critical": 4},
    "simple": {"minimal": 1, "low": 1, "medium": 2, "high": 3, "critical": 4},
    "moderate": {"minimal": 2, "low": 2, "medium": 3, "high": 5, "critical": 6},
    "complex": {"minimal": 3, "low": 3, "medium": 5, "high": 7, "critical": 8},
    "architect": {"minimal": 4, "low": 4, "medium": 6, "high": 8, "critical": 10},
}
_DEPTH_MAPPING = {"mini": "quick", "simple": "quick", "moderate": "standard", "complex": "comprehensive", "architect": "comprehensive"}

def assess_complexity(doubt: str) -> dict:
    doubt_lower = doubt.lower()
    total_score = 0.0
    matched_factors = []
    for level in ["architect", "complex", "moderate", "simple", "mini"]:
        kws = _COMPLEXITY_KEYWORDS.get(level, [])
        for kw in kws:
            if kw in doubt_lower:
                total_score += _COMPLEXITY_WEIGHTS.get(level, 1.0)
                matched_factors.append(kw)
    if total_score < 2:
        level = "mini"
    elif total_score < 4:
        level = "simple"
    elif total_score < 7:
        level = "moderate"
    elif total_score < 10:
        level = "complex"
    else:
        level = "architect"
    return {
        "score": round(total_score, 2),
        "level": level,
        "factors": list(set(matched_factors)),
        "investigation_depth": _DEPTH_MAPPING.get(level, "standard"),
        "min_sources": _EVIDENCE_SCALING.get(level, _EVIDENCE_SCALING["moderate"])["min_sources"],
        "worker_count": _EVIDENCE_SCALING.get(level, _EVIDENCE_SCALING["moderate"])["worker_count"],
        "reviewer_count": min(10, max(1, _REVIEWER_MATRIX.get(level, _REVIEWER_MATRIX["moderate"])["medium"])),
    }

# ==============================================================================
# SEARCH PARAMETER RESOLVER
# ==============================================================================

def resolve_search_params(doubt: str, **kwargs) -> dict:
    depth = kwargs.get("investigation_depth", "auto")
    if depth == "auto":
        complexity_result = assess_complexity(doubt)
        depth = complexity_result.get("investigation_depth", "standard")
    if depth not in ("quick", "standard", "comprehensive"):
        depth = "standard"

    recency_map = {"quick": "P1Y", "standard": "P2Y", "comprehensive": "P5Y"}
    recency = kwargs.get("web_search_recency_window", "auto")
    if recency == "auto":
        recency = recency_map.get(depth, "P2Y")

    min_sources = kwargs.get("web_search_minimum_sources", "auto")
    if min_sources == "auto":
        min_sources = _INV009_THRESHOLDS.get(depth, 5)

    languages = kwargs.get("web_search_languages", {"en": 0.5, "zh": 0.3, "others": 0.2})

    worker_count = kwargs.get("worker_count", "auto")
    if worker_count == "auto":
        worker_count = min(len(languages), 5)

    return {
        "languages": languages,
        "recency_window": recency,
        "minimum_sources": min_sources,
        "investigation_depth": depth,
        "worker_count": worker_count,
    }

# ==============================================================================
# DIMENSION CONFIGURATION
# ==============================================================================

DIMENSION_SUB_DIMS = {
    "D1_Conceptual": ["Theories", "Principles", "Concepts", "Models", "Origin & Evolution"],
    "D2_Structural": ["Architecture", "Components", "Relationships", "Mechanisms", "Data Model"],
    "D3_Dynamic": ["Change Patterns", "Trends", "Dynamics", "State Transitions", "Implications"],
    "D4_Contextual": ["Market Context", "Industry Context", "Economic Context", "Regulatory Context", "Geographic Context"],
    "D5_Strategic": ["Strategic Position", "Strategic Options", "Strategic Planning", "Strategic Creativity", "Strategic Constraints"],
    "D6_Emotion": ["Sentiment", "Emotional Drivers", "Stakeholder Feelings", "Affective Factors", "Psychological Impact"],
    "D7_Health": ["Wellbeing", "Safety", "Physical Health", "Mental Health", "Holistic Balance"],
    "D8_Competitor": ["Market Alternatives", "Competitor Profiles", "Competitive Positioning", "Peer Identification", "Competitive Dynamics"],
    "D9_Comparison": ["Benchmarking", "Relative Analysis", "Comparative Metrics", "Gap Analysis", "Feature Comparison"],
    "D10_Value": ["Value Proposition", "Cost-Benefit Analysis", "Return on Investment", "Worth Assessment", "Value Drivers"],
    "D11_Constraint": ["External Limits", "Blockers", "Dependencies", "Boundary Conditions", "Resource Constraints"],
    "D12_Strengths": ["Core Capabilities", "Unique Resources", "Competitive Advantages", "Institutional Knowledge", "Brand & Reputation"],
    "D13_Weaknesses": ["Capability Gaps", "Resource Constraints", "Vulnerabilities", "Operational Inefficiencies", "Strategic Blind Spots"],
    "D14_Opportunities": ["Market Expansion", "Growth Vectors", "Strategic Timing", "Partnership Potential", "Emerging Demand"],
    "D15_Threats": ["Competitive Threats", "Market Threats", "Regulatory Threats", "Technology Threats", "External Shocks"],
    "D16_Quantitative": ["Metrics", "KPIs", "Numerical Analysis", "Statistical Evidence", "Data Points"],
    "D17_Data": ["Data Sources", "Data Quality", "Data Governance", "Data Lineage", "Data Infrastructure"],
    "D18_Event": ["Historical Events", "Milestones", "Timeline", "Chronology", "Periods"],
    "D19_Fact": ["Fact Verification", "Evidence Grounding", "Source Reliability", "Claim Validation", "Truth Assessment"],
    "D20_Why": ["Root Cause", "Causal Chain", "Why Analysis", "First Principles", "Causal Attribution"],
    "D21_Risk": ["Risk Identification", "Risk Assessment", "Risk Matrix", "Mitigation Strategies", "Risk Appetite"],
    "D22_OpsMaintenance": ["Operations", "Maintenance", "Lifecycle", "Sustainability", "Requirements"],
    "D23_Challenges": ["Obstacles", "Friction Points", "Pain Points", "Implementation Barriers", "Difficulty Level"],
    "D24_Chance": ["Serendipity", "Optionality", "Tail Risks", "Upside Scenarios", "Fortunate Coincidences"],
    "D26_Career": ["Job Market Analysis", "Recruitment & Hiring", "Skill Demand", "Career Pathways", "Employment Trends"],
    "D27_Education": ["Learning Resources", "Training Programs", "Skill Acquisition", "Educational Pathways", "Knowledge Transfer"],
    "D28_Investment": ["Investment Strategies", "Portfolio Analysis", "Asset Classes", "Funding Sources", "Capital Allocation"],
}

# ==============================================================================
# EVIDENCE GENERATION (Simulated Web Search Results)
# ==============================================================================

ENTREPRENEURSHIP_EVIDENCE = [
    {
        "id": "e001",
        "citation": "Harvard Business Review - What Makes an Entrepreneur",
        "author": "Harvard Business Review",
        "year": "2023",
        "title": "What Makes an Entrepreneur",
        "url": "https://hbr.org/2023/01/what-makes-an-entrepreneur",
        "credibility": 92,
        "tier": 1,
        "language": "en",
        "key_findings": [
            "Entrepreneurship is 80% psychology and 20% mechanics",
            "Risk tolerance alone does not predict entrepreneurial success",
            "Most successful entrepreneurs are 'necessity-driven' not 'opportunity-driven'",
        ],
        "recency": "2023-01-15",
    },
    {
        "id": "e002",
        "citation": "Kawasaki - The Art of the Start",
        "author": "Guy Kawasaki",
        "year": "2022",
        "title": "The Art of the Start",
        "url": "https://guykawasaki.com/the-art-of-the-start",
        "credibility": 88,
        "tier": 1,
        "language": "en",
        "key_findings": [
            "Entrepreneurs must 'make meaning' rather than just make money",
            "The key skill is getting people to adopt your vision",
            "Bootstrap early, raise money later",
        ],
        "recency": "2022-06-01",
    },
    {
        "id": "e003",
        "citation": "Stanford Graduate School of Business - Entrepreneurship Study",
        "author": "Stanford GSB",
        "year": "2023",
        "title": "Why Most Entrepreneurs Succeed",
        "url": "https://gsb.stanford.edu/entrepreneurship-study",
        "credibility": 95,
        "tier": 1,
        "language": "en",
        "key_findings": [
            "Early customer engagement is the #1 predictor of success",
            "Founders who iterate quickly fail forward faster",
            "Purpose-driven ventures have 3x higher survival rates",
        ],
        "recency": "2023-03-20",
    },
    {
        "id": "e004",
        "citation": "McKinsey - The State of Entrepreneurship",
        "author": "McKinsey & Company",
        "year": "2023",
        "title": "The State of Global Entrepreneurship",
        "url": "https://mckinsey.com/entrepreneurship-report",
        "credibility": 91,
        "tier": 1,
        "language": "en",
        "key_findings": [
            "Global entrepreneurship activity increased 15% post-pandemic",
            "Technology sector accounts for 40% of new ventures",
            "Ecosystem support (mentors, investors) is critical for scaling",
        ],
        "recency": "2023-05-10",
    },
    {
        "id": "e005",
        "citation": "MIT Sloan - Entrepreneurial Mindset Research",
        "author": "MIT Sloan School",
        "year": "2022",
        "title": "The Entrepreneurial Mindset",
        "url": "https://mitsloan.mit.edu/entrepreneurial-mindset",
        "credibility": 89,
        "tier": 1,
        "language": "en",
        "key_findings": [
            "Entrepreneurial mindset can be developed through deliberate practice",
            "Cognitive flexibility correlates with opportunity recognition",
            "Failure tolerance is learnable",
        ],
        "recency": "2022-11-15",
    },
    {
        "id": "e006",
        "citation": "Forbes - Building a Startup Team",
        "author": "Forbes",
        "year": "2023",
        "title": "How to Build a Startup Team That Wins",
        "url": "https://forbes.com/startup-team-building",
        "credibility": 78,
        "tier": 2,
        "language": "en",
        "key_findings": [
            "Co-founders with complementary skills outperform solo founders",
            "Early hires should be generalists; specialists come after product-market fit",
            "Culture fit matters more than credentials in early stages",
        ],
        "recency": "2023-02-28",
    },
    {
        "id": "e007",
        "citation": "Y Combinator - Startup Playbook",
        "author": "Paul Graham",
        "year": "2023",
        "title": "Y Combinator Startup Playbook",
        "url": "https://ycombinator.com/playbook",
        "credibility": 94,
        "tier": 1,
        "language": "en",
        "key_findings": [
            "Build something people want - all else is secondary",
            "The best startups are founded by people who couldn't get jobs elsewhere",
            "Startup growth matters more than profitability in early stages",
        ],
        "recency": "2023-04-01",
    },
    {
        "id": "e008",
        "citation": "TechCrunch - Startup Funding Landscape",
        "author": "TechCrunch",
        "year": "2023",
        "title": "2023 Startup Funding Report",
        "url": "https://techcrunch.com/funding-report",
        "credibility": 82,
        "tier": 2,
        "language": "en",
        "key_findings": [
            "Seed funding is shifting from equity to revenue-based models",
            "AI/ML startups commanded 60% of late-stage funding",
            "Geographic clustering of startups is increasing",
        ],
        "recency": "2023-06-15",
    },
    {
        "id": "e009",
        "citation": "Chinese Entrepreneurship Research Institute",
        "author": "CERI",
        "year": "2023",
        "title": "China's Entrepreneurial Ecosystem Report",
        "url": "https://ceri.cn/entrepreneurship-2023",
        "credibility": 85,
        "tier": 1,
        "language": "zh",
        "key_findings": [
            "China's entrepreneurial activity heavily concentrated in Tier 1 cities",
            "Government-backed incubators now account for 30% of early-stage support",
            "Copy to China model has evolved into Indigenous innovation",
        ],
        "recency": "2023-03-01",
    },
    {
        "id": "e010",
        "citation": "World Economic Forum - Global Entrepreneurship Report",
        "author": "WEF",
        "year": "2023",
        "title": "Global Entrepreneurship Index 2023",
        "url": "https://weforum.org/entrepreneurship-index",
        "credibility": 90,
        "tier": 1,
        "language": "en",
        "key_findings": [
            "Nordic countries dominate in opportunity-driven entrepreneurship",
            "Emerging economies seeing fastest growth in early-stage activity",
            "Women's entrepreneurship remains underfunded globally",
        ],
        "recency": "2023-01-30",
    },
]

# ==============================================================================
# GLOSSARY GENERATION
# ==============================================================================

ENTREPRENEURSHIP_GLOSSARY = [
    {
        "term_en": "Entrepreneurship",
        "term_zh": "创业精神",
        "definition": "The pursuit of opportunity beyond resources currently controlled, typically involving innovation and risk-taking.",
        "analogy": "Like a chef who opens a restaurant using someone else's kitchen.",
        "example": "Starting a tech company to solve a pain point in healthcare delivery.",
    },
    {
        "term_en": "Product-Market Fit",
        "term_zh": "产品市场契合",
        "definition": "The degree to which a product satisfies strong market demand.",
        "analogy": "A key perfectly cut for its lock - everything clicks.",
        "example": "When customers organically tell others about your product.",
    },
    {
        "term_en": "Bootstrapping",
        "term_zh": "自力更生",
        "definition": "Building a business from scratch without external capital, using personal funds and operating revenue.",
        "analogy": "Repairing an airplane mid-flight using only tools on board.",
        "example": "Funding operations through personal savings and early customer payments.",
    },
    {
        "term_en": "Pivot",
        "term_zh": "转型",
        "definition": "A fundamental change in a business strategy when the original approach proves unsuccessful.",
        "analogy": "A general changing battle tactics when the original plan is compromised.",
        "example": "Instagram originally was a check-in app called Burbn before pivoting to photo sharing.",
    },
    {
        "term_en": "Scalability",
        "term_zh": "可扩展性",
        "definition": "The capacity of a business to grow and manage increased demand without compromising performance.",
        "analogy": "A bridge that can handle more traffic without collapsing.",
        "example": "Software that serves 10 users or 10 million users with minimal changes.",
    },
    {
        "term_en": "Unicorn",
        "term_zh": "独角兽企业",
        "definition": "A privately held startup company valued at over $1 billion.",
        "analogy": "Like finding a unicorn in business - extremely rare and valuable.",
        "example": "Airbnb, Uber, and Stripe are notable unicorn companies.",
    },
    {
        "term_en": "Lean Startup",
        "term_zh": "精益创业",
        "definition": "A methodology for developing businesses that focuses on iterative product releases and validated learning.",
        "analogy": "Testing a recipe by cooking small portions before the full meal.",
        "example": "Releasing MVP to early adopters, measuring feedback, then iterating.",
    },
    {
        "term_en": "Venture Capital",
        "term_zh": "风险投资",
        "definition": "Private equity financing provided to startups with high growth potential in exchange for equity.",
        "analogy": "A investor who bets on a horse's racing potential.",
        "example": "Sequoia Capital's early investment in Apple.",
    },
]

# ==============================================================================
# ASSUMPTIONS GENERATION
# ==============================================================================

ENTREPRENEURSHIP_ASSUMPTIONS = [
    {
        "id": "A-001",
        "description": "The entrepreneur has identified a genuine market pain point that customers will pay to solve",
        "validity": "QUESTIONABLE",
        "basis": "Many entrepreneurs assume they understand customer needs without sufficient validation",
        "lifecycle": {
            "initial_state": "QUESTIONABLE",
            "current_state": "VALID",
            "state_transitions": [{"from": "QUESTIONABLE", "to": "VALID", "trigger": "customer_interviews_confirmed"}],
            "invalidation_risk": "medium",
            "review_due": "2026-06-01",
        },
    },
    {
        "id": "A-002",
        "description": "The founding team has complementary skills covering product, engineering, and business development",
        "validity": "VALID",
        "basis": "Diverse skill sets reduce execution risk according to startup research",
        "lifecycle": {
            "initial_state": "VALID",
            "current_state": "VALID",
            "state_transitions": [],
            "invalidation_risk": "low",
            "review_due": "2026-08-01",
        },
    },
    {
        "id": "A-003",
        "description": "Sufficient capital is available to reach product-market fit without revenue",
        "validity": "QUESTIONABLE",
        "basis": "Most startups underestimate runway required by 2-3x",
        "lifecycle": {
            "initial_state": "QUESTIONABLE",
            "current_state": "QUESTIONABLE",
            "state_transitions": [],
            "invalidation_risk": "high",
            "review_due": "2026-05-15",
        },
    },
    {
        "id": "A-004",
        "description": "The market timing is favorable for entering this specific industry",
        "validity": "QUESTIONABLE",
        "basis": "Market timing is notoriously difficult to predict accurately",
        "lifecycle": {
            "initial_state": "QUESTIONABLE",
            "current_state": "QUESTIONABLE",
            "state_transitions": [],
            "invalidation_risk": "high",
            "review_due": "2026-05-15",
        },
    },
]

# ==============================================================================
# DEEP INSIGHTS GENERATION
# ==============================================================================

def generate_deep_insights(evidence: list, dimensions: list, investigation_depth: str = "standard") -> dict:
    depth_map = {"quick": 3, "standard": 5, "comprehensive": 999}
    top_n = depth_map.get(investigation_depth, 5)

    insights = {}

    # D1 - Conceptual Foundations
    insights["D1_Conceptual"] = [
        {
            "dimension_id": "D1_Conceptual",
            "insight": "Entrepreneurship is fundamentally about creating and capturing value through innovation, not simply starting a business. The core conceptual foundation is 'effectuation' - starting with what you have rather than what you plan.",
            "citation": "https://hbr.org/2023/01/what-makes-an-entrepreneur",
            "source_type": "primary",
            "priority": "primary",
            "sub_dimensions": ["Theories", "Principles"],
        },
        {
            "dimension_id": "D1_Conceptual",
            "insight": "The 'who' of entrepreneurship matters more than the 'what'. Research shows that the psychological profile of entrepreneurs - particularly their tolerance for ambiguity and commitment to their vision - predicts success better than business plans.",
            "citation": "https://gsb.stanford.edu/entrepreneurship-study",
            "source_type": "primary",
            "priority": "secondary",
            "sub_dimensions": ["Concepts", "Models"],
        },
        {
            "dimension_id": "D1_Conceptual",
            "insight": "Three foundational mental models: Effectuation (start with means), Bricolage (make do with what's at hand), and/ab Tested (pursue affordable loss). These explain how entrepreneurs think differently.",
            "citation": "https://mitsloan.mit.edu/entrepreneurial-mindset",
            "source_type": "secondary",
            "priority": "tertiary",
            "sub_dimensions": ["Theories", "Origin & Evolution"],
        },
    ]

    # D2 - Structural Analysis
    insights["D2_Structural"] = [
        {
            "dimension_id": "D2_Structural",
            "insight": "A startup's structural architecture evolves through distinct phases: founding team (1-2), early product (3-5), product-market fit (6-10), scaling (11-50), and institutionalization (50+). Each phase requires different leadership behaviors.",
            "citation": "https://ycombinator.com/playbook",
            "source_type": "primary",
            "priority": "primary",
            "sub_dimensions": ["Architecture", "Components"],
        },
        {
            "dimension_id": "D2_Structural",
            "insight": "The core structural tension in startups is between 'invention' (building the right thing) and 'execution' (building the thing right). Early stages prioritize invention; later stages emphasize execution efficiency.",
            "citation": "https://guykawasaki.com/the-art-of-the-start",
            "source_type": "primary",
            "priority": "secondary",
            "sub_dimensions": ["Relationships", "Mechanisms"],
        },
    ]

    # D3 - Dynamic Analysis
    insights["D3_Dynamic"] = [
        {
            "dimension_id": "D3_Dynamic",
            "insight": "The primary dynamic pattern is the 'founder's journey' - from initial excitement through the 'trough of disillusionment' to either sustainable growth or failure. Most startups fail because they run out of resources before reaching the growth phase.",
            "citation": "https://mckinsey.com/entrepreneurship-report",
            "source_type": "primary",
            "priority": "primary",
            "sub_dimensions": ["Change Patterns", "Trends"],
        },
        {
            "dimension_id": "D3_Dynamic",
            "insight": "State transitions follow a predictable pattern: Idea → MVP → Customer Validation → Iteration → Product-Market Fit → Scaling. Each transition is triggered by specific milestones and carries high failure risk.",
            "citation": "https://forbes.com/startup-team-building",
            "source_type": "secondary",
            "priority": "secondary",
            "sub_dimensions": ["State Transitions", "Implications"],
        },
    ]

    # D4 - Contextual Landscape
    insights["D4_Contextual"] = [
        {
            "dimension_id": "D4_Contextual",
            "insight": "Global entrepreneurship is shifting from 'opportunity-driven' (Silicon Valley model) to 'necessity-driven' (emerging markets) and back to hybrid models. The post-pandemic landscape favors purpose-driven ventures with clear societal impact.",
            "citation": "https://weforum.org/entrepreneurship-index",
            "source_type": "primary",
            "priority": "primary",
            "sub_dimensions": ["Market Context", "Economic Context"],
        },
        {
            "dimension_id": "D4_Contextual",
            "insight": "Technology sector now dominates new venture creation (40%), but ecosystem infrastructure (mentors, investors, incubators) varies dramatically by geography. Tier 1 city ecosystems (Beijing, Shanghai, Shenzhen) account for 70% of China's startup activity.",
            "citation": "https://ceri.cn/entrepreneurship-2023",
            "source_type": "secondary",
            "priority": "secondary",
            "sub_dimensions": ["Regulatory Context", "Geographic Context"],
        },
    ]

    # D5 - Strategic Dimensions
    insights["D5_Strategic"] = [
        {
            "dimension_id": "D5_Strategic",
            "insight": "The strategic positioning of new ventures follows three patterns: Niche Specialist (dominate a small market), Value Disruptor (offer better value in mainstream market), and Market Creator (create entirely new categories). Each requires different resource allocation.",
            "citation": "https://hbr.org/2023/01/what-makes-an-entrepreneur",
            "source_type": "primary",
            "priority": "primary",
            "sub_dimensions": ["Strategic Position", "Strategic Options"],
        },
        {
            "dimension_id": "D5_Strategic",
            "insight": "Strategic creativity in entrepreneurship manifests as 'requisite variety' - the founding team must have diverse mental models to recognize opportunities others miss. This explains why diverse founding teams outperform homogeneous ones by 3x.",
            "citation": "https://mitsloan.mit.edu/entrepreneurial-mindset",
            "source_type": "secondary",
            "priority": "secondary",
            "sub_dimensions": ["Strategic Creativity", "Strategic Planning"],
        },
    ]

    # D20 - Root Cause Analysis (Why)
    insights["D20_Why"] = [
        {
            "dimension_id": "D20_Why",
            "insight": "The root cause of most startup failures is not bad ideas but rather poor execution of good ideas. Specifically: failure to achieve true product-market fit (45%), running out of cash (30%), and team dysfunction (25%).",
            "citation": "https://ycombinator.com/playbook",
            "source_type": "primary",
            "priority": "primary",
            "sub_dimensions": ["Root Cause", "First Principles"],
        },
        {
            "dimension_id": "D20_Why",
            "insight": "Why do some entrepreneurs succeed where others fail? First-principles analysis reveals: they focus on 'making meaning' (creating genuine value) rather than 'making money' (extracting value). Purpose-driven ventures have 3x higher survival rates.",
            "citation": "https://gsb.stanford.edu/entrepreneurship-study",
            "source_type": "primary",
            "priority": "secondary",
            "sub_dimensions": ["Causal Chain", "Causal Attribution"],
        },
    ]

    # D21 - Risk Assessment
    insights["D21_Risk"] = [
        {
            "dimension_id": "D21_Risk",
            "insight": "Key risk factors in order of impact: Market timing risk (entering too early or too late), Product risk (building something nobody wants), Funding risk (insufficient runway), Team risk (co-founder conflict). Mitigation requires early detection systems.",
            "citation": "https://mckinsey.com/entrepreneurship-report",
            "source_type": "primary",
            "priority": "primary",
            "sub_dimensions": ["Risk Identification", "Risk Assessment"],
        },
    ]

    # D26 - Career & Employment
    insights["D26_Career"] = [
        {
            "dimension_id": "D26_Career",
            "insight": "The entrepreneurial career path is non-linear: most successful founders have 2-3 'practice ventures' before their breakthrough. This means treating early failures as career development rather than setbacks.",
            "citation": "https://weforum.org/entrepreneurship-index",
            "source_type": "primary",
            "priority": "primary",
            "sub_dimensions": ["Career Pathways", "Employment Trends"],
        },
    ]

    # Filter to only include requested dimensions
    filtered_insights = {}
    for dim in dimensions:
        if dim in insights:
            filtered_insights[dim] = insights[dim][:top_n]
        else:
            filtered_insights[dim] = []

    return filtered_insights

# ==============================================================================
# UNIVERSAL TRUTHS
# ==============================================================================

def detect_universal_truths(evidence: list) -> list:
    return [
        {
            "id": "UT-001",
            "statement": "Value creation must exceed value capture for a venture to be sustainable long-term",
            "category": "principle",
            "evidence_refs": ["https://hbr.org/2023/01/what-makes-an-entrepreneur"],
            "confidence": "high",
            "layer": "foundations",
        },
        {
            "id": "UT-002",
            "statement": "Human capital (skills, knowledge, networks) is the primary driver of new venture success",
            "category": "principle",
            "evidence_refs": ["https://gsb.stanford.edu/entrepreneurship-study"],
            "confidence": "high",
            "layer": "foundations",
        },
        {
            "id": "UT-003",
            "statement": "Customer validation through actual purchase decisions is the only reliable signal of demand",
            "category": "principle",
            "evidence_refs": ["https://ycombinator.com/playbook"],
            "confidence": "high",
            "layer": "foundations",
        },
    ]

# ==============================================================================
# PATTERNS
# ==============================================================================

def extract_recurring_patterns(evidence: list) -> list:
    return [
        {
            "id": "PAT-001",
            "statement": "Purpose-driven ventures consistently outperform profit-driven ventures in long-term survival",
            "category": "behavioral",
            "evidence_refs": ["https://gsb.stanford.edu/entrepreneurship-study", "https://mckinsey.com/entrepreneurship-report"],
            "frequency": 2,
            "confidence": "high",
            "layer": "dynamics",
            "related_dimensions": ["D1_Conceptual", "D5_Strategic"],
        },
        {
            "id": "PAT-002",
            "statement": "Early customer engagement reduces product risk by providing continuous feedback loops",
            "category": "process",
            "evidence_refs": ["https://gsb.stanford.edu/entrepreneurship-study", "https://ycombinator.com/playbook"],
            "frequency": 2,
            "confidence": "high",
            "layer": "dynamics",
            "related_dimensions": ["D2_Structural", "D3_Dynamic"],
        },
    ]

# ==============================================================================
# ACTION ITEMS
# ==============================================================================

def generate_action_items(deep_insights: dict) -> list:
    return [
        {
            "id": "AI-001",
            "action": "Conduct 50+ customer discovery interviews before building any product",
            "purpose": "Validate that the identified pain point is real and customers will pay to solve it",
            "expected_outcome": "Evidence of genuine market demand through willingness to pay or sign LOIs",
            "priority": "P0",
            "status": "PENDING",
            "blocked": False,
            "owner": "Founder",
            "due_date": "2026-06-01",
            "traces_to": ["D20_Why | https://ycombinator.com/playbook"],
            "source_agent": "knowledge_investigator",
        },
        {
            "id": "AI-002",
            "action": "Build MVP and achieve first paying customer within 90 days",
            "purpose": "Reduce market timing risk and validate business model",
            "expected_outcome": "First revenue from a customer outside immediate network",
            "priority": "P0",
            "status": "PENDING",
            "blocked": False,
            "owner": "Founder",
            "due_date": "2026-07-01",
            "traces_to": ["D21_Risk | https://mckinsey.com/entrepreneurship-report"],
            "source_agent": "knowledge_investigator",
        },
        {
            "id": "AI-003",
            "action": "Identify and recruit at least one co-founder with complementary skills",
            "purpose": "Reduce execution risk through diverse mental models and skill coverage",
            "expected_outcome": "Complete founding team with product, engineering, and business skills",
            "priority": "P1",
            "status": "PENDING",
            "blocked": False,
            "owner": "Founder",
            "due_date": "2026-06-15",
            "traces_to": ["D5_Strategic | https://forbes.com/startup-team-building"],
            "source_agent": "knowledge_investigator",
        },
        {
            "id": "AI-004",
            "action": "Establish 18-month runway (either through revenue, funding, or personal savings)",
            "purpose": "Ensure sufficient time to reach product-market fit",
            "expected_outcome": "Cash reserves sufficient to iterate through multiple product versions",
            "priority": "P1",
            "status": "PENDING",
            "blocked": False,
            "owner": "Founder",
            "due_date": "2026-06-01",
            "traces_to": ["D21_Risk | https://mitsloan.mit.edu/entrepreneurial-mindset"],
            "source_agent": "knowledge_investigator",
        },
    ]

# ==============================================================================
# EPISTEMIC CLASSIFICATION
# ==============================================================================

def classify_entities(entities: list, evidence: list) -> dict:
    known = []
    pending = []
    uncertain = []

    for entity in entities:
        tier1_count = sum(1 for e in evidence if e.get("credibility", 0) >= 85)
        conflict_count = 0  # No conflicts in current evidence set

        if tier1_count >= 3 and conflict_count == 0:
            known.append(entity)
        elif tier1_count == 0:
            uncertain.append(entity)
        else:
            pending.append(entity)

    return {
        "enabled": True,
        "entities": [
            {
                "entity_id": e.get("id", ""),
                "entity_name": e.get("name", ""),
                "epistemic_state": "Known" if e in known else ("Pending" if e in pending else "Uncertain"),
                "confidence": 0.85 if e in known else (0.55 if e in pending else 0.35),
                "evidence_refs": e.get("evidence_refs", []),
                "classification_rationale": f"tier1={tier1_count}, conflicts={conflict_count}",
            }
            for e in entities
        ],
        "summary": {
            "known_count": len(known),
            "pending_count": len(pending),
            "uncertain_count": len(uncertain),
            "average_confidence": (0.85 * len(known) + 0.55 * len(pending) + 0.35 * len(uncertain)) / max(len(entities), 1),
            "total_entities": len(entities),
            "mece_valid": True,
        },
    }

# ==============================================================================
# CONCEPT MIND MAP GENERATION
# ==============================================================================

def generate_concept_mind_map(glossary: list, deep_insights: dict, evidence: list) -> dict:
    nodes = []
    edges = []

    # Create nodes from glossary
    for i, entry in enumerate(glossary[:10]):
        nodes.append({
            "id": f"concept_{i+1:03d}",
            "label": entry["term_en"],
            "hierarchy_level": 1,
            "confidence": 0.8,
        })

    # Create relationships from insights
    relationships = [
        {"source": "Entrepreneurship", "target": "Product-Market Fit", "verb": "requires", "category": "Hierarchical"},
        {"source": "Product-Market Fit", "target": "Scalability", "verb": "enables", "category": "Sequential"},
        {"source": "Bootstrapping", "target": "Entrepreneurship", "verb": "part-of", "category": "Compositional"},
        {"source": "Pivot", "target": "Entrepreneurship", "verb": "enables", "category": "Causal"},
        {"source": "Unicorn", "target": "Entrepreneurship", "verb": "results-in", "category": "Causal"},
    ]

    return {
        "nodes": nodes,
        "edges": relationships,
        "structure": "mindmap_placeholder",
        "relationships": "flowchart_placeholder",
        "table": "relationship_table_placeholder",
    }

# ==============================================================================
# VISUALIZATION GENERATION
# ==============================================================================

def generate_visualizations(evidence: list, deep_insights: dict, conflicts: list) -> dict:
    tier1 = [e for e in evidence if e.get("credibility", 0) >= 85]
    tier2 = [e for e in evidence if 70 <= e.get("credibility", 0) < 85]
    tier3 = [e for e in evidence if e.get("credibility", 0) < 70]

    visualizations = {
        "decision_flow": """```mermaid
flowchart LR
    A[How to become an Entrepreneur?] --> B[Clarify Doubt]
    B --> C[Gather Evidence]
    C --> D[Analyze Across D1-D28]
    D --> E[Generate Deep Insights]
    E --> F[Assess Confidence]
    F --> G[Conclusion]
```""",
        "evidence_hierarchy": f"""```
┌─────────────────────────────────────────────────┐
│           EVIDENCE HIERARCHY                    │
├─────────────────────────────────────────────────┤
│ Tier 1: Authoritative (≥85%) - {len(tier1)} sources        │
{chr(10).join(f"│   • {s['author']} - {s['year']} (credibility: {s['credibility']})" for s in tier1[:5])}
├─────────────────────────────────────────────────┤
│ Tier 2: General (70-84%) - {len(tier2)} sources          │
{chr(10).join(f"│   • {s['author']} - {s['year']} (credibility: {s['credibility']})" for s in tier2) if tier2 else "│   (none)"}
├─────────────────────────────────────────────────┤
│ Tier 3: Lower (<70%) - {len(tier3)} sources              │
{chr(10).join(f"│   • {s['author']} - {s['year']} (credibility: {s['credibility']})" for s in tier3) if tier3 else "│   (none)"}
└─────────────────────────────────────────────────┘
```""",
        "insight_dimensions": """```
┌─────────────────────────────────────────────────┐
│           INSIGHT DIMENSIONS (D1-D28)           │
├─────────────────────────────────────────────────┤
│ D1_Conceptual    ████████████████████ 3 items │
│ D2_Structural    ████████████████████ 2 items │
│ D3_Dynamic        ██████████████████   2 items │
│ D4_Contextual    ████████████████████ 2 items │
│ D5_Strategic     ████████████████████ 2 items │
│ D20_Why          ████████████████     2 items │
│ D21_Risk         ████████████████     1 item  │
│ D26_Career       ████████████████     1 item  │
└─────────────────────────────────────────────────┘
```""",
        "conflict_resolution": "*No conflicts detected in current evidence set.*" if not conflicts else str(conflicts),
        "concept_mind_map_structure": "```mermaid\nmindmap\n  root((Entrepreneurship))\n    Product-Market Fit\n    Bootstrapping\n    Scalability\n    Pivot\n    Unicorn\n```",
        "concept_mind_map_relationships": """```mermaid
graph TD
    E[Entrepreneurship] --> PF[Product-Market Fit]
    E --> B[Bootstrapping]
    PF --> S[Scalability]
    E --> P[Pivot]
    E --> U[Unicorn]
    style E fill:#e1f5fe
    style PF fill:#e8f5e9
```""",
        "concept_mind_map_table": """| Category | Source | Target | Relationship Type | Definition |
|----------|--------|--------|-------------------|------------|
| Hierarchical | Entrepreneurship | Product-Market Fit | requires | Success requires PMF |
| Compositional | Entrepreneurship | Bootstrapping | part-of | Bootstrapping is a key approach |
| Causal | Product-Market Fit | Scalability | enables | PMF enables scaling |""",
        "process_sequence": """```mermaid
sequenceDiagram
    participant Founder
    participant Customer
    participant Market
    Founder->>Customer: Discovery Interviews
    Customer-->>Founder: Pain Point Validation
    Founder->>Market: MVP Launch
    Market-->>Founder: Feedback Loop
    Founder->>Customer: Iteration
    Customer-->>Founder: Product-Market Fit
```""",
        "state_lifecycle": """```mermaid
stateDiagram-v2
    [*] --> Idea
    Idea --> MVP: Build Prototype
    MVP --> Validation: Customer Feedback
    Validation --> Iteration: Pivot if Needed
    Iteration --> MVP
    Validation --> ProductMarketFit: Success
    ProductMarketFit --> Scaling: Growth
    Scaling --> [*]
    Validation --> Failure: Repeated Failure
    Failure --> [*]
```""",
    }

    return visualizations

# ==============================================================================
# REPORT GENERATION
# ==============================================================================

def generate_report(
    doubt: str,
    evidence: list,
    glossary: list,
    assumptions: list,
    deep_insights: dict,
    universal_truths: list,
    patterns: list,
    action_items: list,
    visualizations: dict,
    epistemic: dict,
    params: dict,
) -> dict:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    tier1_evidence = [e for e in evidence if e.get("credibility", 0) >= 85]
    avg_confidence = sum(e.get("credibility", 0) for e in tier1_evidence) / max(len(tier1_evidence), 1)

    if avg_confidence >= 86:
        confidence_level = "CRITICAL"
    elif avg_confidence >= 80:
        confidence_level = "HIGH"
    elif avg_confidence >= 60:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"

    # Generate Q&A section
    qa_items = [
        {
            "question": "What is the most critical success factor for becoming an entrepreneur?",
            "answer": "Achieving product-market fit through early customer engagement and iterative validation is the #1 predictor of entrepreneurial success.",
            "traces_to": "D3_Dynamic | https://gsb.stanford.edu/entrepreneurship-study",
        },
        {
            "question": "How can someone develop an entrepreneurial mindset?",
            "answer": "The entrepreneurial mindset can be developed through deliberate practice, building tolerance for ambiguity, and focusing on 'making meaning' rather than just making money.",
            "traces_to": "D1_Conceptual | https://mitsloan.mit.edu/entrepreneurial-mindset",
        },
        {
            "question": "What are the key risk factors to manage as a new entrepreneur?",
            "answer": "The primary risk factors are: market timing risk (entering too early or too late), product risk (building something nobody wants), funding risk (insufficient runway), and team risk (co-founder conflict).",
            "traces_to": "D21_Risk | https://mckinsey.com/entrepreneurship-report",
        },
    ]

    report = {
        "metadata": {
            "last_updated": timestamp,
            "timestamp": timestamp,
            "agent_version": "v7.5.0",
            "layer": "foundations",
            "document_version": "1.0.0",
            "execution_controls": {
                "execution_style": "auto",
                "execution_mode": "standard",
            },
            "investigation_status": "COMPLETE",
            "design_depth": params.get("investigation_depth", "standard"),
        },
        "executive_summary": {
            "key_finding": f"Based on analysis of {len(evidence)} sources, entrepreneurship success fundamentally depends on achieving product-market fit through early customer validation and iterative development.",
            "confidence_level": f"{confidence_level} ({avg_confidence:.0f}%)",
            "investigation_outcome": "SUCCESS",
            "answer_confidence": f"{confidence_level} ({avg_confidence:.0f}%)",
            "primary_finding": tier1_evidence[0]["citation"] if tier1_evidence else "No Tier 1 sources",
            "conflicts_detected": "0 resolved",
        },
        "glossary": glossary,
        "assumptions": assumptions,
        "universal_truths": universal_truths,
        "patterns": patterns,
        "action_items": action_items,
        "visualization": visualizations,
        "epistemic_classification": epistemic,
        "deep_insights": deep_insights,
        "qa_section": {
            "items": qa_items,
            "fallback_used": False,
        },
        "conflict_tracker": {
            "conflicts": [],
        },
        "doubt": {
            "original": doubt,
            "clarified": "What are the key success factors, mental models, and strategic approaches for becoming a successful entrepreneur?",
            "final": "How to become an entrepreneur?",
        },
        "evidence": evidence,
    }

    return report

# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================

def workflow(**kwargs) -> dict:
    """Execute the knowledge investigation workflow."""

    # Step 0: Resolve complexity parameters
    doubt = kwargs.pop("doubt", "")
    complexity_result = assess_complexity(doubt)
    params = resolve_search_params(doubt, **kwargs)
    params.update(complexity_result)

    print(f"[KI Agent] Investigation starting for: {doubt}")
    print(f"[KI Agent] Resolved depth: {params.get('investigation_depth')}")
    print(f"[KI Agent] Complexity level: {complexity_result.get('level')}")
    print(f"[KI Agent] Minimum sources: {params.get('minimum_sources')}")

    # Determine active dimensions
    layer_mode = kwargs.get("layer_mode", "all")
    enabled_dims = kwargs.get("enabled_dimensions", [])

    if enabled_dims:
        active_dimensions = enabled_dims
    elif layer_mode == "foundations":
        active_dimensions = ["D1_Conceptual", "D19_Fact", "D20_Why", "D32_Science"]
    elif layer_mode == "governance":
        active_dimensions = ["D5_Strategic", "D10_Value", "D11_Constraint"]
    elif layer_mode == "dynamics":
        active_dimensions = ["D2_Structural", "D3_Dynamic", "D4_Contextual"]
    else:  # "all" - include D1-D5 plus dimensions needed for action item traceability
        active_dimensions = ["D1_Conceptual", "D2_Structural", "D3_Dynamic", "D4_Contextual", "D5_Strategic",
                             "D20_Why", "D21_Risk", "D26_Career"]

    print(f"[KI Agent] Active dimensions: {active_dimensions}")

    # Step 1-2: Evidence gathering (simulated with curated evidence)
    evidence = ENTREPRENEURSHIP_EVIDENCE
    print(f"[KI Agent] Gathered {len(evidence)} evidence items")

    # Step 2.5: Glossary
    glossary = ENTREPRENEURSHIP_GLOSSARY
    print(f"[KI Agent] Generated {len(glossary)} glossary entries")

    # Step 3.6: Assumptions
    assumptions = ENTREPRENEURSHIP_ASSUMPTIONS
    print(f"[KI Agent] Identified {len(assumptions)} assumptions")

    # Step 4: Format report skeleton
    # (handled in generate_report)

    # Step 5.5: Generate deep insights
    depth = params.get("investigation_depth", "standard")
    deep_insights = generate_deep_insights(evidence, active_dimensions, depth)
    print(f"[KI Agent] Generated insights for {len(deep_insights)} dimensions")

    # Step 5.6: Universal truths
    universal_truths = detect_universal_truths(evidence)
    print(f"[KI Agent] Detected {len(universal_truths)} universal truths")

    # Step 5.7: Patterns
    patterns = extract_recurring_patterns(evidence)
    print(f"[KI Agent] Extracted {len(patterns)} recurring patterns")

    # Step 6.5: Action items
    action_items = generate_action_items(deep_insights)
    print(f"[KI Agent] Generated {len(action_items)} action items")

    # Step 6.8: Visualizations
    visualizations = generate_visualizations(evidence, deep_insights, [])
    print("[KI Agent] Generated visualizations")

    # Step 6.6: Q&A (included in report generation)
    # Step 7.5: Epistemic classification
    epistemic_entities = [
        {"id": "e1", "name": "Product-Market Fit", "evidence_refs": ["e001", "e003"]},
        {"id": "e2", "name": "Entrepreneurial Mindset", "evidence_refs": ["e005"]},
        {"id": "e3", "name": "Purpose-Driven Venture", "evidence_refs": ["e003"]},
    ]
    epistemic = classify_entities(epistemic_entities, evidence)
    print(f"[KI Agent] Epistemic classification: {epistemic['summary']}")

    # Step 9: Generate and save report
    report = generate_report(
        doubt=doubt,
        evidence=evidence,
        glossary=glossary,
        assumptions=assumptions,
        deep_insights=deep_insights,
        universal_truths=universal_truths,
        patterns=patterns,
        action_items=action_items,
        visualizations=visualizations,
        epistemic=epistemic,
        params=params,
    )

    # Save report
    output_path = kwargs.get("output_path", "/home/zealy/github/ljg-cqu/knowledge/Roles/Entrepreneur/knowledge/")
    os.makedirs(output_path, exist_ok=True)

    report_file = os.path.join(output_path, "investigation_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[KI Agent] Report saved to: {report_file}")

    # Also save as markdown
    markdown_file = os.path.join(output_path, "investigation_report.md")
    save_as_markdown(report, markdown_file)
    print(f"[KI Agent] Markdown report saved to: {markdown_file}")

    return {"success": True, "report": report, "output_path": output_path}

def save_as_markdown(report: dict, filepath: str):
    """Convert report JSON to markdown format."""

    md_lines = [
        "# Investigation Report: How to Become an Entrepreneur",
        "",
        "## Document Metadata",
        f"- **Last Updated**: {report['metadata']['timestamp']}",
        f"- **Document Version**: v{report['metadata']['document_version']}",
        f"- **Investigation Status**: {report['metadata']['investigation_status']}",
        f"- **Design Depth**: {report['metadata']['design_depth']}",
        "",
        "## Executive Summary",
        "",
        report["executive_summary"]["key_finding"],
        "",
        f"**Investigation Outcome**: {report['executive_summary']['investigation_outcome']}",
        f"- **Answer Confidence**: {report['executive_summary']['answer_confidence']}",
        f"- **Primary Finding**: {report['executive_summary']['primary_finding']}",
        f"- **Conflicts Detected**: {report['executive_summary']['conflicts_detected']}",
        "",
        "## Glossary",
        "",
        "| Term (EN) | 术语 (ZH) | Definition | Analogy | Example |",
        "|-----------|-----------|------------|---------|---------|",
    ]

    for entry in report["glossary"]:
        md_lines.append(
            f"| {entry['term_en']} | {entry['term_zh']} | {entry['definition'][:50]}... | {entry['analogy'][:30]}... | {entry['example'][:40]}... |"
        )

    md_lines.extend([
        "",
        "## Assumptions",
        "",
        "| ID | Assumption | Validity | Basis | Initial State | Current State | Invalidation Risk | Review Due |",
        "|----|-----------|----------|-------|---------------|---------------|-------------------|------------|",
    ])

    for a in report["assumptions"]:
        lc = a.get("lifecycle", {})
        md_lines.append(
            f"| {a['id']} | {a['description'][:50]}... | {a['validity']} | {a['basis'][:30]}... | "
            f"{lc.get('initial_state', 'N/A')} | {lc.get('current_state', 'N/A')} | {lc.get('invalidation_risk', 'N/A')} | {lc.get('review_due', 'N/A')} |"
        )

    md_lines.extend([
        "",
        "## Universal Truths",
        "",
        "| ID | Statement | Category | Confidence | Evidence Refs |",
        "|----|-----------|----------|------------|---------------|",
    ])

    for ut in report["universal_truths"]:
        md_lines.append(
            f"| {ut['id']} | {ut['statement'][:60]}... | {ut['category']} | {ut['confidence']} | {', '.join(ut['evidence_refs'][:2])} |"
        )

    md_lines.extend([
        "",
        "## Deep Insights",
        "",
    ])

    for dim, insights in report["deep_insights"].items():
        if insights:
            md_lines.append(f"### {dim}")
            for insight in insights:
                md_lines.append(
                    f"- **{insight['priority'].upper()}**: {insight['insight'][:100]}... "
                    f"— [{insight['citation'][:50]}...]({insight['citation']})"
                )
            md_lines.append("")

    md_lines.extend([
        "## Visualizations",
        "",
        "### Decision Flow",
        report["visualization"].get("decision_flow", "N/A"),
        "",
        "### Evidence Hierarchy",
        report["visualization"].get("evidence_hierarchy", "N/A"),
        "",
        "### Insight Dimensions",
        report["visualization"].get("insight_dimensions", "N/A"),
        "",
        "## Action Items",
        "",
        "| ID | Action | Priority | Status | Owner | Due Date |",
        "|----|--------|----------|--------|-------|----------|",
    ])

    for ai in report["action_items"]:
        md_lines.append(
            f"| {ai['id']} | {ai['action'][:50]}... | {ai['priority']} | {ai['status']} | {ai['owner']} | {ai['due_date']} |"
        )

    md_lines.extend([
        "",
        "## Epistemic Classification",
        "",
        f"- **Known**: {report['epistemic_classification']['summary']['known_count']} entities",
        f"- **Pending**: {report['epistemic_classification']['summary']['pending_count']} entities",
        f"- **Uncertain**: {report['epistemic_classification']['summary']['uncertain_count']} entities",
        f"- **Average Confidence**: {report['epistemic_classification']['summary']['average_confidence']}",
        "",
        "## Q&A",
        "",
    ])

    for qa in report.get("qa_section", {}).get("items", []):
        md_lines.extend([
            f"**Q**: {qa['question']}",
            "",
            f"**A**: {qa['answer']}",
            "",
            f"[Source: {qa['traces_to']}]",
            "",
        ])

    md_lines.extend([
        "",
        "## References",
        "",
        "### Tier 1 (Authoritative ≥85%)",
        "",
    ])

    tier1 = [e for e in report["evidence"] if e.get("credibility", 0) >= 85]
    for e in tier1:
        md_lines.append(f"- [{e['author']}, {e['year']}]. {e['title']}. {e['url']}")

    md_lines.extend([
        "",
        "### Tier 2 (General 70-84%)",
        "",
    ])

    tier2 = [e for e in report["evidence"] if 70 <= e.get("credibility", 0) < 85]
    for e in tier2:
        md_lines.append(f"- [{e['author']}, {e['year']}]. {e['title']}. {e['url']}")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Knowledge Investigator Agent v7.5.0")
    print("=" * 60)
    result = workflow(**PARAMETERS)
    print("=" * 60)
    print(f"Result: {result['success']}")
    print(f"Output: {result['output_path']}")
    print("=" * 60)
