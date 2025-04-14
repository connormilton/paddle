# grasshooper.py - Advanced collaborative LLM forex trading system

import os
import json
import time
import logging
import pandas as pd
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import openai
from trading_ig import IGService
from polygon import RESTClient

# --- Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CollaborativeTrader")
os.makedirs("data", exist_ok=True)

# Core currency pairs to trade
FOREX_PAIRS = [
    "CS.D.EURUSD.TODAY.IP", "CS.D.USDJPY.TODAY.IP", "CS.D.GBPUSD.TODAY.IP", 
    "CS.D.AUDUSD.TODAY.IP", "CS.D.USDCAD.TODAY.IP", "CS.D.GBPJPY.TODAY.IP",
    "CS.D.EURJPY.TODAY.IP", "CS.D.AUDJPY.TODAY.IP", "CS.D.EURGBP.TODAY.IP",
    "CS.D.USDCHF.TODAY.IP", "CS.D.NZDUSD.TODAY.IP", "CS.D.AUDNZD.TODAY.IP"
]

# Trading prompt templates
class CollaborativeTradingPrompts:
    """Advanced prompt templates for a team of 3 collaborative trading agents"""
    
    @staticmethod
    def market_scanner(market_data, account_data, positions, recent_trades, trade_logs=None, agent_feedback=None):
        """
        Build prompt for market scanner agent (Scout)
        The scout identifies opportunities and maintains market awareness
        """
        # Format market data summary
        market_summary = ""
        for epic, data in market_data.items():
            if "current" in data:
                current = data["current"]
                market_summary += f"\n{epic}: Bid/Ask: {current.get('bid')}/{current.get('offer')}"
        
        # Format positions
        positions_info = "No open positions"
        open_position_count = 0
        if not positions.empty:
            positions_info = positions.to_string()
            open_position_count = len(positions)
        
        # Format recent trades
        recent_trades_text = "No recent trades"
        if recent_trades:
            recent_trades_text = "\n".join([
                f"- {t.get('timestamp', '')[:10]} {t.get('epic')} {t.get('direction')}: {t.get('outcome')} " +
                f"(Reason: {t.get('pattern', 'Unknown pattern')})"
                for t in recent_trades
            ])
        
        # Get feedback from other agents
        feedback_section = ""
        if agent_feedback:
            feedback_section = "\n## Recent Agent Feedback\n"
            for agent, feedback in agent_feedback.items():
                feedback_section += f"### {agent}\n{feedback}\n"
        
        # Trade log insights
        trade_log_insights = ""
        if trade_logs:
            # Extract patterns that worked well
            successful_patterns = {}
            failed_patterns = {}
            for trade in trade_logs:
                pattern = trade.get("pattern", "Unknown")
                outcome = trade.get("outcome", "").upper()
                
                if "WIN" in outcome or "PROFIT" in outcome:
                    successful_patterns[pattern] = successful_patterns.get(pattern, 0) + 1
                elif "LOSS" in outcome or "STOPPED" in outcome:
                    failed_patterns[pattern] = failed_patterns.get(pattern, 0) + 1
            
            # Format insights
            trade_log_insights = "\n## Trade Log Insights\n"
            
            # Successful patterns
            trade_log_insights += "### Successful Patterns\n"
            for pattern, count in sorted(successful_patterns.items(), key=lambda x: x[1], reverse=True)[:3]:
                trade_log_insights += f"- {pattern}: {count} successful trades\n"
            
            # Failed patterns
            trade_log_insights += "\n### Challenging Patterns\n"
            for pattern, count in sorted(failed_patterns.items(), key=lambda x: x[1], reverse=True)[:3]:
                trade_log_insights += f"- {pattern}: {count} failed trades\n"
        
        # Create the main part of the prompt with variables
        main_prompt = f"""
# Forex Market Scout Agent

## Your Role and Goals
You are the Market Scout in a team of three collaborative trading agents targeting 10% daily returns.
Your specific responsibilities are:
1. Identify promising trading opportunities across forex pairs
2. Maintain broad market awareness and detect correlations
3. Recommend at least 5-7 high-potential opportunities

## Current Trading Status
- Account Balance: {account_data.get('balance')}
- Available Funds: {account_data.get('available')}
- Open Positions: {open_position_count} (Target minimum: 3)
- Daily Return Target: 10% of account

## Open Positions
{positions_info}

## Recent Trading History
{recent_trades_text}
{trade_log_insights}
{feedback_section}

## Current Market Prices
{market_summary}

## Your Task
1. Analyze all forex pairs to identify 5-7 high-potential trading opportunities
2. Consider correlation between currencies and overall market conditions
3. Prioritize finding setups with clear price action patterns
4. Evaluate how your suggestions complement existing positions
5. If we have fewer than 3 open positions, prioritize finding new opportunities

## Self-Improvement
Reflect on past trade performance from the logs. How can you improve your opportunity identification? What patterns have been working? What information would help you make better recommendations?
"""
        
        # Use raw string for response format to avoid f-string formatting issues
        response_format = r"""
## Response Format
Respond with a JSON object containing:
1. "market_assessment" with:
   - "overall_condition": "trending/ranging/volatile/uncertain"
   - "overall_bias": "bullish/bearish/neutral"
   - "correlations": [key currency correlations you've identified]
   - "summary": "Brief 1-2 sentence market summary"

2. "opportunities" array with the best trading setups:
   - "epic": Currency pair code
   - "pattern": Specific pattern identified
   - "direction": "BUY" or "SELL"
   - "conviction": Rating from 1-10
   - "timeframe": "short-term/medium-term/long-term"
   - "reasoning": Detailed explanation of the opportunity
   - "key_levels": Important price levels to watch

3. "self_improvement":
   - "pattern_effectiveness": Assessment of which patterns have worked best
   - "questions_for_team": Questions for other agents about what information would help you
   - "suggestions": Ideas to improve the overall trading system
"""

        # Combine main prompt with response format
        return main_prompt + response_format

    @staticmethod
    def analysis_engine(opportunities, market_data, account_data, positions, system_memory, previous_analyses=None):
        """
        Build prompt for analysis agent (Strategist)
        The strategist performs deep technical analysis and develops trading plans
        """
        # Format selected opportunities
        opps_section = ""
        for opp in opportunities:
            epic = opp.get("epic")
            if epic in market_data:
                data = market_data[epic]
                
                # Format price data
                price_info = ""
                if "h1" in data and data["h1"]:
                    candles = data["h1"][-5:]  # Last 5 candles
                    for candle in reversed(candles):
                        dt = datetime.fromisoformat(candle.get('timestamp', ''))
                        time_str = dt.strftime("%H:%M")
                        price_info += f"\n- {time_str}: O={candle.get('open'):.5f} H={candle.get('high'):.5f} L={candle.get('low'):.5f} C={candle.get('close'):.5f}"
                
                opps_section += f"""
## {epic}
- Pattern: {opp.get('pattern')}
- Direction: {opp.get('direction')}
- Conviction: {opp.get('conviction')}/10
- Reasoning: {opp.get('reasoning')}
- Key Levels: {opp.get('key_levels', 'None specified')}

### Recent Price Data:
{price_info}
"""
        
        # Format positions
        positions_info = "No open positions"
        open_position_count = 0
        if not positions.empty:
            positions_info = positions.to_string()
            open_position_count = len(positions)
        
        # Previous analyses insights
        previous_analyses_section = ""
        if previous_analyses:
            previous_analyses_section = "\n## Previous Analysis Insights\n"
            for pair, analyses in previous_analyses.items():
                if analyses:
                    latest = analyses[-1]
                    previous_analyses_section += f"### {pair}\n"
                    previous_analyses_section += f"- Last Analysis: {latest.get('timestamp', 'Unknown')}\n"
                    previous_analyses_section += f"- Direction: {latest.get('direction', 'Unknown')}\n"
                    previous_analyses_section += f"- Outcome: {latest.get('outcome', 'Pending')}\n"
                    previous_analyses_section += f"- Key Insight: {latest.get('key_insight', 'None')}\n"
        
        # Create the main part of the prompt with variables
        main_prompt = f"""
# Forex Strategist Agent

## Your Role and Goals
You are the Strategic Analyst in a team of three collaborative trading agents targeting 10% daily returns.
Your specific responsibilities are:
1. Perform detailed technical analysis on promising opportunities
2. Determine precise entry/exit points and stop loss levels
3. Create specific trading plans with clear risk management
4. Calculate risk-reward ratios and evaluate trade quality

## Current Trading Status
- Account Balance: {account_data.get('balance')}
- Available Funds: {account_data.get('available')}
- Open Positions: {open_position_count} (Target minimum: 3)
- Daily Return Target: 10% of account
- Win Rate: {system_memory.get('win_count', 0)}/{system_memory.get('trade_count', 0)} trades

## Open Positions
{positions_info}

{previous_analyses_section}

## Selected Opportunities
{opps_section}

## Your Task
1. Analyze each opportunity and determine if it meets your quality criteria
2. For each valid opportunity, develop a complete trading plan with:
   - Precise entry zone with ideal price and acceptable range
   - Multiple take profit targets based on key levels
   - Stop loss level with clear reasoning
   - Position sizing recommendation based on risk
3. Calculate and evaluate risk-reward ratios (aim for 1:2 minimum)
4. If we have fewer than 3 open positions, prioritize finding executable trades

## Self-Improvement
Reflect on your analysis techniques. What additional information or tools would help you create better trade plans? What patterns have had the highest success rate? How can you improve your stop placement?
"""
        
        # Use raw string for response format to avoid f-string formatting issues
        response_format = r"""
## Response Format
Respond with a JSON object containing:
1. "analysis_results" array with detailed analysis for each pair:
   - "epic": Currency pair code
   - "direction": "BUY" or "SELL"
   - "analysis_quality": Rating from 1-10
   - "entry_zone": {"ideal": 1.2345, "range_low": 1.2340, "range_high": 1.2350}
   - "stop_loss": {"price": 1.2300, "pips": 45, "reasoning": "Below key support"}
   - "take_profit": [{"level": 1.2400, "pips": 55, "probability": 70}]
   - "risk_reward": Calculated R:R ratio
   - "position_size_recommendation": Percentage of account or lot size
   - "trading_plan": Detailed explanation of the setup and execution
   - "key_indicators": Important technical indicators supporting the trade

2. "market_insights": Overall insights from your analysis

3. "self_improvement":
   - "analysis_effectiveness": Assessment of your previous analyses
   - "questions_for_team": Questions for other agents that would help you
   - "suggestions": Ideas to improve your analysis approach
"""

        # Combine main prompt with response format
        return main_prompt + response_format

    @staticmethod
    def decision_maker(analysis_results, account_data, positions, system_memory, market_data, 
                       recent_trades=None, execution_history=None):
        """
        Build prompt for decision agent (Executor)
        The executor makes final decisions and manages overall portfolio risk
        """
        # Format analysis results
        analysis_section = ""
        for result in analysis_results:
            epic = result.get("epic")
            
            # Format take profit levels
            take_profit_text = ""
            for idx, tp in enumerate(result.get("take_profit", [])):
                take_profit_text += f"TP{idx+1}: {tp.get('level')} ({tp.get('pips')} pips, {tp.get('probability')}% prob)\n      "
            
            analysis_section += f"""
## {epic}
- Direction: {result.get('direction')}
- Analysis Quality: {result.get('analysis_quality')}/10
- Entry Zone: {result.get('entry_zone', {}).get('ideal')} (Range: {result.get('entry_zone', {}).get('range_low')} - {result.get('entry_zone', {}).get('range_high')})
- Stop Loss: {result.get('stop_loss', {}).get('price')} ({result.get('stop_loss', {}).get('pips')} pips)
- Take Profit: {take_profit_text}
- Risk-Reward: {result.get('risk_reward')}
- Size Recommendation: {result.get('position_size_recommendation')}
- Trading Plan: {result.get('trading_plan')}
"""
        
        # Format positions
        positions_info = "No open positions"
        open_position_count = 0
        if not positions.empty:
            positions_info = positions.to_string()
            open_position_count = len(positions)
        
        # Format recent trades
        recent_trades_text = "No recent trades"
        if recent_trades:
            recent_trades_text = "\n".join([
                f"- {t.get('timestamp', '')[:10]} {t.get('epic')} {t.get('direction')}: {t.get('outcome')} " +
                f"(Risk: {t.get('risk_percent', 'Unknown')}%, R:R: {t.get('risk_reward', 'Unknown')})"
                for t in recent_trades
            ])
        
        # Get system performance metrics
        win_rate = 0
        if system_memory.get('trade_count', 0) > 0:
            win_rate = (system_memory.get('win_count', 0) / system_memory.get('trade_count', 0)) * 100
        
        # Execution history insights
        execution_insights = ""
        if execution_history:
            # Extract patterns that worked well
            position_sizes = []
            risk_percents = []
            risk_rewards = []
            
            for trade in execution_history:
                if "size" in trade:
                    position_sizes.append(float(trade["size"]))
                if "risk_percent" in trade:
                    risk_percents.append(float(trade["risk_percent"]))
                if "risk_reward" in trade:
                    risk_rewards.append(float(trade["risk_reward"]))
            
            # Calculate averages
            avg_size = sum(position_sizes) / len(position_sizes) if position_sizes else 0
            avg_risk = sum(risk_percents) / len(risk_percents) if risk_percents else 0
            avg_rr = sum(risk_rewards) / len(risk_rewards) if risk_rewards else 0
            
            execution_insights = f"""
## Execution History Insights
- Average Position Size: {avg_size:.2f}
- Average Risk per Trade: {avg_risk:.2f}%
- Average Risk-Reward: {avg_rr:.2f}
"""
        
        # Calculate daily progress toward 10% goal
        daily_return = system_memory.get("daily_return", 0)
        daily_goal = 10.0  # 10% daily target
        
        # Current market conditions (brief)
        market_conditions = ""
        for epic, data in list(market_data.items())[:5]:  # Just show first 5 pairs
            if "current" in data:
                current = data["current"]
                market_conditions += f"\n- {epic}: {current.get('bid')}/{current.get('offer')}"
        
        # Create the main part of the prompt with variables
        main_prompt = f"""
# Forex Executor Agent

## Your Role and Goals
You are the Decision Executor in a team of three collaborative trading agents targeting 10% daily returns.
Your specific responsibilities are:
1. Make final trading decisions based on analysis
2. Manage overall portfolio risk and exposure
3. Ensure we maintain at least 3 open positions at all times
4. Balance risk across currency pairs and correlations
5. Implement sophisticated stop management strategies

## Current Trading Status
- Account Balance: {account_data.get('balance')}
- Available Funds: {account_data.get('available')}
- Open Positions: {open_position_count} (Target minimum: 3)
- Daily Return Target: 10% of account (Current progress: {daily_return:.2f}%)
- Win Rate: {win_rate:.1f}% ({system_memory.get('win_count', 0)}/{system_memory.get('trade_count', 0)} trades)
- Risk Multiplier: {system_memory.get('risk_multiplier', 1.0)}x

## Current Market Snapshot
{market_conditions}

## Open Positions
{positions_info}

## Recent Trading History
{recent_trades_text}
{execution_insights}

## Analysis Results
{analysis_section}

## Your Task
1. Decide which trade opportunities to execute based on quality and portfolio balance
2. Determine final position sizes and risk levels (aim for consistent risk per trade)
3. If we have fewer than 3 open positions, prioritize opening new positions
4. For existing positions, decide if any need to be closed or have stops adjusted
5. Implement risk management strategies for each new position

## Stop Management Strategies
1. FIXED STOP - Traditional fixed stop loss placement
2. TRAILING STOP - Stop follows price at a fixed distance as it moves in your favor
3. BREAKEVEN STOP - Move stop to entry once trade has moved a certain distance
4. PARTIAL PROFIT - Take profits on part of position at first target

## Self-Improvement
Reflect on execution performance. What position sizing has worked best? Which stop strategies have been most effective? How can you better balance the portfolio to achieve the 10% daily target?
"""
        
        # Use raw string for response format to avoid f-string formatting issues
        response_format = r"""
## Response Format
Respond with a JSON object containing:
1. "trade_actions" array with specific trades to execute:
   - "action_type": "OPEN"
   - "epic": Currency pair code
   - "direction": "BUY" or "SELL"
   - "size": Position size
   - "entry_price": Ideal entry price
   - "entry_range": [lower, upper] range for entry
   - "initial_stop_loss": Initial stop loss level
   - "take_profit_levels": [level1, level2, ...]
   - "risk_percent": Percentage of account risked
   - "risk_reward": Expected R:R ratio
   - "pattern": Pattern being traded
   - "stop_management": [{"type": "BREAKEVEN_STOP", "settings": {"activation_pips": 20}}]
   - "reasoning": Detailed reasoning for this execution decision

2. "position_actions" array with actions for existing positions:
   - "action_type": "CLOSE", "UPDATE_STOP", "TAKE_PARTIAL"
   - "epic": Currency pair code
   - "dealId": Deal identifier
   - "new_level": New level for stop or take profit
   - "percentage": Percentage to close (for partial)
   - "reason": Explanation for the action

3. "portfolio_assessment":
   - "current_exposure": Overall market exposure assessment
   - "risk_distribution": How risk is spread across positions
   - "correlation_management": How you're managing correlations
   - "progress_to_daily_goal": Assessment of progress toward 10% target

4. "self_improvement":
   - "execution_effectiveness": Assessment of your decision making
   - "questions_for_team": Questions for other agents that would help you
   - "suggestions": Ideas to improve execution approach
   - "needs_from_user": What you need from the human trader to improve
"""

        # Combine main prompt with response format
        return main_prompt + response_format

    @staticmethod
    def team_review(agent_responses, system_memory, market_data, positions, account_data, daily_perf=None):
        """
        Build prompt for team review and coordination
        This allows the agents to review each other's work and coordinate strategy
        """
        # Format agent responses
        agent_summaries = ""
        for agent, response in agent_responses.items():
            if agent == "scout":
                opportunities = response.get("opportunities", [])
                opps_summary = f"Found {len(opportunities)} opportunities.\n"
                for opp in opportunities[:3]:  # Top 3
                    opps_summary += f"- {opp.get('epic')}: {opp.get('direction')} ({opp.get('conviction')}/10) - {opp.get('pattern')}\n"
                
                agent_summaries += f"""
## Scout's Findings
- Market Assessment: {response.get('market_assessment', {}).get('overall_condition', 'Unknown')} / {response.get('market_assessment', {}).get('overall_bias', 'Unknown')}
- Top Opportunities:
{opps_summary}
- Self-improvement notes: {response.get('self_improvement', {}).get('suggestions', 'None provided')}
"""
            
            elif agent == "strategist":
                analyses = response.get("analysis_results", [])
                analyses_summary = f"Analyzed {len(analyses)} opportunities.\n"
                for analysis in analyses[:3]:  # Top 3
                    analyses_summary += f"- {analysis.get('epic')}: {analysis.get('direction')} (Quality: {analysis.get('analysis_quality')}/10) - R:R {analysis.get('risk_reward')}\n"
                
                agent_summaries += f"""
## Strategist's Analyses
- Key insights: {response.get('market_insights', 'None provided')}
- Top Analyses:
{analyses_summary}
- Self-improvement notes: {response.get('self_improvement', {}).get('suggestions', 'None provided')}
"""
            
            elif agent == "executor":
                trades = response.get("trade_actions", [])
                positions = response.get("position_actions", [])
                trades_summary = f"Executed {len(trades)} trades, modified {len(positions)} positions.\n"
                for trade in trades[:3]:  # Top 3
                    trades_summary += f"- {trade.get('epic')}: {trade.get('direction')} (Risk: {trade.get('risk_percent')}%, R:R: {trade.get('risk_reward')})\n"
                
                agent_summaries += f"""
## Executor's Decisions
- Portfolio assessment: {response.get('portfolio_assessment', {}).get('progress_to_daily_goal', 'Not provided')}
- Actions:
{trades_summary}
- Self-improvement notes: {response.get('self_improvement', {}).get('suggestions', 'None provided')}
"""
        
        # Format daily performance if available
        performance_section = ""
        if daily_perf:
            profit_loss = daily_perf.get('profit_loss', 0)
            return_pct = daily_perf.get('return_percent', 0)
            win_trades = daily_perf.get('winning_trades', 0)
            loss_trades = daily_perf.get('losing_trades', 0)
            
            performance_section = f"""
## Daily Performance Review
- Profit/Loss: {profit_loss}
- Return: {return_pct:.2f}% (Target: 10%)
- Winning Trades: {win_trades}
- Losing Trades: {loss_trades}
- Win Rate: {(win_trades / (win_trades + loss_trades) * 100) if (win_trades + loss_trades) > 0 else 0:.1f}%
"""
        
        # Create the main part of the prompt with variables
        main_prompt = f"""
# Trading Team Review Session

## Current Status
- Account Balance: {account_data.get('balance')}
- Open Positions: {len(positions) if not positions.empty else 0} (Target minimum: 3)
- Win Rate: {(system_memory.get('win_count', 0) / system_memory.get('trade_count', 1)) * 100:.1f}% 
- Risk Multiplier: {system_memory.get('risk_multiplier', 1.0)}x

{performance_section}

## Agent Summaries
{agent_summaries}

## Team Review Questions
1. How are the agents coordinating effectively? What gaps exist in communication?
2. Are we maintaining the minimum of 3 open positions at all times?
3. What strategies are working best toward our 10% daily target?
4. How can each agent better support the others?
5. What information or tools do we need from our human operator?
"""
        
        # Use raw string for response format to avoid f-string formatting issues
        response_format = r"""
## Response Format
Respond with a JSON object containing:
1. "team_assessment":
   - "coordination_quality": Rating from 1-10
   - "progress_to_goal": Assessment of progress toward 10% daily target
   - "key_strengths": What's working well
   - "key_weaknesses": Areas needing improvement

2. "agent_feedback":
   - "scout": Specific feedback and advice for the scout
   - "strategist": Specific feedback and advice for the strategist
   - "executor": Specific feedback and advice for the executor

3. "strategy_adjustments":
   - "risk_management": Suggested changes to risk approach
   - "pair_selection": Suggested changes to market focus
   - "technical_approach": Suggested changes to analysis methods

4. "requests_for_human":
   - Specific questions or requests for the human operator
"""

        # Combine main prompt with response format
        return main_prompt + response_format

    @staticmethod
    def position_manager(positions, market_data, system_memory, execution_history=None):
        """
        Build prompt for position management
        Focuses on managing existing positions rather than opening new ones
        """
        # Format positions
        positions_info = "No open positions"
        if not positions.empty:
            positions_info = positions.to_string()
        
        # Format current market data for relevant pairs
        market_info = ""
        for _, position in positions.iterrows():
            epic = position.get("epic")
            if epic in market_data and "current" in market_data[epic]:
                current = market_data[epic]["current"]
                direction = position.get("direction")
                opening_level = position.get("level")
                current_level = current.get("bid") if direction == "SELL" else current.get("offer")
                
                # Calculate pip movement and profit/loss
                multiplier = 0.01 if "JPY" in epic else 0.0001
                pips_moved = (current_level - opening_level) / multiplier
                if direction == "SELL":
                    pips_moved = -pips_moved
                
                market_info += f"\n## {epic}\n"
                market_info += f"- Current Price: {current.get('bid')}/{current.get('offer')}\n"
                market_info += f"- Opening Level: {opening_level}\n"
                market_info += f"- Movement: {pips_moved:.1f} pips\n"
                
                # Add recent price action
                if "h1" in market_data[epic] and market_data[epic]["h1"]:
                    last_candle = market_data[epic]["h1"][-1]
                    market_info += f"- Latest H1: O={last_candle.get('open'):.5f} H={last_candle.get('high'):.5f} L={last_candle.get('low'):.5f} C={last_candle.get('close'):.5f}\n"
        
        # Position management history
        management_history = ""
        if execution_history:
            updates = [t for t in execution_history if t.get("action_type") in ["UPDATE_STOP", "TAKE_PARTIAL", "CLOSE"]]
            if updates:
                management_history = "\n## Recent Position Management Actions\n"
                for update in updates[:5]:
                    management_history += f"- {update.get('timestamp', '')[:10]} {update.get('epic')} {update.get('action_type')}: {update.get('reason')}\n"
        
        # Create the main part of the prompt with variables
        main_prompt = f"""
# Position Management Agent

## Your Role and Goals
You specialize in managing existing trading positions to:
1. Protect capital by moving stops to breakeven when appropriate
2. Maximize profit by implementing trailing stops
3. Take partial profits at key levels
4. Close positions when technical conditions change

## Current Trading Status
- Account Balance: {system_memory.get('balance')}
- Open Positions: {len(positions) if not positions.empty else 0}
- Win Rate: {(system_memory.get('win_count', 0) / system_memory.get('trade_count', 1)) * 100:.1f}%
- Daily Return Target: 10% of account

## Open Positions
{positions_info}

## Current Market Data
{market_info}

{management_history}

## Your Task
Analyze each open position and determine:
1. If stop loss needs to be adjusted (breakeven, tightened, or trailed)
2. If partial profits should be taken at current levels
3. If any position should be completely closed due to changing conditions
4. The optimal balance between securing profit and allowing room for further movement

## Position Management Strategies
1. BREAKEVEN STOP - Move stop to entry once trade has moved a certain distance in your favor
2. TRAILING STOP - Stop follows price at a fixed distance as it moves in your favor
3. PARTIAL PROFIT - Take profits on part of position at current level
4. FULL CLOSE - Close entire position
5. NO CHANGE - Leave position as is
"""
        
        # Use raw string for response format to avoid f-string formatting issues
        response_format = r"""
## Response Format
Respond with a JSON object containing:
1. "position_actions" array with actions for each position:
   - "action_type": "CLOSE", "UPDATE_STOP", "TAKE_PARTIAL", "NO_CHANGE"
   - "epic": Currency pair code
   - "dealId": Deal identifier
   - "new_level": New level for stop (if applicable)
   - "percentage": Percentage to close (if applicable)
   - "reason": Detailed reasoning for this action

2. "risk_assessment":
   - "overall_portfolio_risk": Assessment of current risk level
   - "primary_threats": Main risks to current positions
   - "recommended_adjustments": General portfolio adjustment recommendations
"""

        # Combine main prompt with response format
        return main_prompt + response_format

class LLMBudgetManager:
    """Lightweight manager for LLM API budget"""
    
    def __init__(self):
        self.daily_budget = float(os.getenv("DAILY_LLM_BUDGET", 20.0))
        self.usage_file = "data/usage_log.jsonl"
        self.today = datetime.now(timezone.utc).date().isoformat()
        self.refresh_usage()
        
    def refresh_usage(self):
        """Load or initialize today's usage data"""
        os.makedirs(os.path.dirname(self.usage_file), exist_ok=True)
        
        self.usage = {"date": self.today, "total_cost": 0.0, "calls": []}
        
        if os.path.exists(self.usage_file):
            try:
                with open(self.usage_file, "r") as f:
                    for line in f:
                        entry = json.loads(line)
                        if entry.get("date") == self.today:
                            self.usage = entry
                            break
            except:
                pass

    def log_usage(self, tier, tokens_in, tokens_out, cost):
        """Log LLM API usage"""
        self.usage["total_cost"] += cost
        self.usage["calls"].append({
            "tier": tier,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost": cost,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Save to file
        entries = []
        if os.path.exists(self.usage_file):
            with open(self.usage_file, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("date") != self.today:
                            entries.append(entry)
                    except:
                        continue
        
        entries.append(self.usage)
        
        with open(self.usage_file, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
                
        return cost
    
    def can_spend(self, estimated_cost):
        """Check if we have enough budget remaining"""
        return self.usage["total_cost"] + estimated_cost <= self.daily_budget
    
    def get_status(self):
        """Get current budget status"""
        return {
            "total_budget": self.daily_budget,
            "spent": self.usage["total_cost"],
            "remaining": self.daily_budget - self.usage["total_cost"],
            "percent_used": (self.usage["total_cost"] / self.daily_budget) * 100
        }

class TradingMemory:
    """Enhanced memory system for collaborative trading"""
    
    def __init__(self):
        self.memory_file = "data/system_memory.json"
        self.feedback_file = "data/agent_feedback.json"
        self.trade_log_file = "data/trade_log.jsonl"
        self.analysis_file = "data/analysis_history.json"
        
        # Initialize all memory systems
        self.load_memory()
        self.load_feedback()
        self.load_analysis_history()
    
    def load_memory(self):
        """Load or initialize system memory"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    self.memory = json.load(f)
            except:
                self.initialize_memory()
        else:
            self.initialize_memory()
    
    def load_feedback(self):
        """Load or initialize agent feedback"""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, "r") as f:
                    self.feedback = json.load(f)
            except:
                self.initialize_feedback()
        else:
            self.initialize_feedback()
    
    def load_analysis_history(self):
        """Load or initialize analysis history"""
        if os.path.exists(self.analysis_file):
            try:
                with open(self.analysis_file, "r") as f:
                    self.analysis_history = json.load(f)
            except:
                self.initialize_analysis_history()
        else:
            self.initialize_analysis_history()
    
    def initialize_memory(self):
        """Create initial memory structure"""
        self.memory = {
            "created": datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "risk_multiplier": 1.0,
            "base_risk": 1.0,
            "daily_return": 0.0,
            "daily_return_target": 10.0,
            "context": {}
        }
        self.save_memory()
    
    def initialize_feedback(self):
        """Create initial feedback structure"""
        self.feedback = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "scout": {},
            "strategist": {},
            "executor": {}
        }
        self.save_feedback()
    
    def initialize_analysis_history(self):
        """Create initial analysis history structure"""
        self.analysis_history = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "pairs": {}
        }
        self.save_analysis_history()
    
    def save_memory(self):
        """Save memory to disk"""
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f, indent=2)
    
    def save_feedback(self):
        """Save feedback to disk"""
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
        with open(self.feedback_file, "w") as f:
            json.dump(self.feedback, f, indent=2)
    
    def save_analysis_history(self):
        """Save analysis history to disk"""
        os.makedirs(os.path.dirname(self.analysis_file), exist_ok=True)
        with open(self.analysis_file, "w") as f:
            json.dump(self.analysis_history, f, indent=2)
    
    def update_memory(self, key, value):
        """Update a specific memory item"""
        self.memory[key] = value
        self.memory["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.save_memory()
    
    def update_feedback(self, agent, feedback):
        """Update feedback for a specific agent"""
        self.feedback[agent] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content": feedback
        }
        self.feedback["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.save_feedback()
    
    def update_analysis_history(self, pair, analysis):
        """Update analysis history for a specific pair"""
        if pair not in self.analysis_history["pairs"]:
            self.analysis_history["pairs"][pair] = []
        
        # Add new analysis with timestamp
        analysis_entry = analysis.copy()
        analysis_entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Add to history (keep last 10)
        self.analysis_history["pairs"][pair].append(analysis_entry)
        if len(self.analysis_history["pairs"][pair]) > 10:
            self.analysis_history["pairs"][pair] = self.analysis_history["pairs"][pair][-10:]
        
        self.analysis_history["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.save_analysis_history()
    
    def log_trade(self, trade_data):
        """Log a trade and update statistics"""
        # Update basic stats
        self.memory["trade_count"] += 1
        
        outcome = trade_data.get("outcome", "").upper()
        if "WIN" in outcome or "PROFIT" in outcome:
            self.memory["win_count"] += 1
        elif "LOSS" in outcome or "STOPPED" in outcome:
            self.memory["loss_count"] += 1
            
        # Update analysis outcome if this is a close
        if trade_data.get("direction") == "CLOSE" and "epic" in trade_data:
            epic = trade_data.get("epic")
            if epic in self.analysis_history["pairs"] and self.analysis_history["pairs"][epic]:
                # Update the most recent analysis with the outcome
                latest = self.analysis_history["pairs"][epic][-1]
                latest["outcome"] = outcome
                self.save_analysis_history()
        
        # Save trade to log
        with open(self.trade_log_file, "a") as f:
            f.write(json.dumps(trade_data) + "\n")
            
        # Update memory
        self.memory["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.save_memory()
    
    def get_recent_trades(self, limit=5):
        """Get recent trades from log"""
        trades = []
        try:
            if os.path.exists(self.trade_log_file):
                with open(self.trade_log_file, "r") as f:
                    for line in f:
                        trades.append(json.loads(line))
                
                # Sort by timestamp (newest first) and limit
                trades.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                return trades[:limit]
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
        
        return trades
    
    def get_all_trades(self):
        """Get all trades from log"""
        trades = []
        try:
            if os.path.exists(self.trade_log_file):
                with open(self.trade_log_file, "r") as f:
                    for line in f:
                        trades.append(json.loads(line))
                
                # Sort by timestamp (newest first)
                trades.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        except Exception as e:
            logger.error(f"Error getting all trades: {e}")
        
        return trades
    
    def get_agent_feedback(self, agent=None):
        """Get feedback for specific agent or all agents"""
        if agent:
            return self.feedback.get(agent, {})
        else:
            return {
                "scout": self.feedback.get("scout", {}).get("content", "No feedback available"),
                "strategist": self.feedback.get("strategist", {}).get("content", "No feedback available"),
                "executor": self.feedback.get("executor", {}).get("content", "No feedback available")
            }
    
    def get_pair_analysis_history(self, pair=None):
        """Get analysis history for specific pair or all pairs"""
        if pair:
            return self.analysis_history["pairs"].get(pair, [])
        else:
            return self.analysis_history["pairs"]
    
    def calculate_performance_metrics(self):
        """Calculate various performance metrics from trade history"""
        trades = self.get_all_trades()
        
        if not trades:
            return {
                "avg_return_per_trade": 0,
                "avg_risk_per_trade": 0,
                "avg_risk_reward": 0,
                "largest_win": 0,
                "largest_loss": 0
            }
        
        # Extract metrics
        returns = []
        risks = []
        risk_rewards = []
        wins = []
        losses = []
        
        for trade in trades:
            # Skip non-completed trades
            outcome = trade.get("outcome", "").upper()
            if "WIN" not in outcome and "LOSS" not in outcome and "PROFIT" not in outcome and "STOPPED" not in outcome:
                continue
                
            # Extract metrics where available
            if "return_percent" in trade:
                returns.append(float(trade["return_percent"]))
                
                if "WIN" in outcome or "PROFIT" in outcome:
                    wins.append(float(trade["return_percent"]))
                elif "LOSS" in outcome or "STOPPED" in outcome:
                    losses.append(float(trade["return_percent"]))
            
            if "risk_percent" in trade:
                risks.append(float(trade["risk_percent"]))
                
            if "risk_reward" in trade:
                risk_rewards.append(float(trade["risk_reward"]))
        
        # Calculate metrics
        avg_return = sum(returns) / len(returns) if returns else 0
        avg_risk = sum(risks) / len(risks) if risks else 0
        avg_rr = sum(risk_rewards) / len(risk_rewards) if risk_rewards else 0
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0
        
        return {
            "avg_return_per_trade": avg_return,
            "avg_risk_per_trade": avg_risk,
            "avg_risk_reward": avg_rr,
            "largest_win": largest_win,
            "largest_loss": largest_loss
        }

class DataCollector:
    """Collects market and account data"""
    
    def __init__(self, ig_service, polygon_client):
        self.ig = ig_service
        self.polygon = polygon_client
    
    def get_account_data(self):
        """Get account information"""
        try:
            accounts = self.ig.fetch_accounts()
            if os.getenv("IG_ACCOUNT_ID"):
                account = accounts[accounts['accountId'] == os.getenv("IG_ACCOUNT_ID")]
            else:
                account = accounts.iloc[[0]]
            
            return account.iloc[0].to_dict()
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return {}
    
    def get_positions(self):
        """Get open positions"""
        try:
            return self.ig.fetch_open_positions()
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return pd.DataFrame()
    
    def get_market_data(self, epic, timeframes=None):
        """Collect market data for an instrument"""
        if timeframes is None:
            timeframes = {
                "m15": {"timeframe": "15:minute", "lookback_days": 1},
                "h1": {"timeframe": "hour", "lookback_days": 2},
                "h4": {"timeframe": "4:hour", "lookback_days": 5}
            }
        
        # Convert IG epic to Polygon ticker
        ticker_map = {
            "CS.D.EURUSD.TODAY.IP": "C:EURUSD",
            "CS.D.USDJPY.TODAY.IP": "C:USDJPY",
            "CS.D.GBPUSD.TODAY.IP": "C:GBPUSD",
            "CS.D.AUDUSD.TODAY.IP": "C:AUDUSD",
            "CS.D.USDCAD.TODAY.IP": "C:USDCAD",
            "CS.D.USDCHF.TODAY.IP": "C:USDCHF",
            "CS.D.NZDUSD.TODAY.IP": "C:NZDUSD",
            "CS.D.EURJPY.TODAY.IP": "C:EURJPY",
            "CS.D.EURGBP.TODAY.IP": "C:EURGBP",
            "CS.D.GBPJPY.TODAY.IP": "C:GBPJPY",
            "CS.D.AUDJPY.TODAY.IP": "C:AUDJPY",
            "CS.D.AUDNZD.TODAY.IP": "C:AUDNZD"
        }
        
        ticker = ticker_map.get(epic)
        if not ticker:
            return {}
        
        results = {}
        
        try:
            for key, config in timeframes.items():
                timeframe = config["timeframe"]
                lookback_days = config["lookback_days"]
                
                # Parse timeframe
                if ":" in timeframe:
                    parts = timeframe.split(":")
                    multiplier = int(parts[0])
                    timespan = parts[1]
                else:
                    multiplier = 1
                    timespan = timeframe
                
                # Get data from Polygon
                end = datetime.now(timezone.utc)
                start = end - timedelta(days=lookback_days)
                
                aggs = self.polygon.get_aggs(
                    ticker=ticker,
                    multiplier=multiplier,
                    timespan=timespan,
                    from_=start.strftime("%Y-%m-%d"),
                    to=end.strftime("%Y-%m-%d"),
                    limit=100  # Reduced limit to save on data size
                )
                
                if not aggs:
                    continue
                
                # Convert to standardized format
                data = [{
                    "timestamp": datetime.fromtimestamp(a.timestamp/1000, tz=timezone.utc).isoformat(),
                    "open": a.open,
                    "high": a.high,
                    "low": a.low,
                    "close": a.close,
                    "volume": a.volume
                } for a in aggs]
                
                results[key] = data
            
            # Add current price snapshot
            snapshot = self.get_price_snapshot(epic)
            if snapshot:
                results["current"] = snapshot
                
            return results
        except Exception as e:
            logger.error(f"Error collecting market data for {epic}: {e}")
            return {}
    
    def get_price_snapshot(self, epic):
        """Get current market price snapshot"""
        try:
            response = self.ig.fetch_market_by_epic(epic)
            if response and 'snapshot' in response:
                snapshot = response['snapshot']
                
                # Get raw values
                raw_bid = snapshot.get('bid')
                raw_offer = snapshot.get('offer')
                
                # Determine the divisor based on the currency pair
                divisor = 100.0 if "JPY" in epic else 10000.0
                
                # Convert points to decimal format
                bid = raw_bid / divisor if raw_bid is not None else None
                offer = raw_offer / divisor if raw_offer is not None else None
                
                return {
                    "bid": bid,
                    "offer": offer,
                    "epic": epic,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            return None
        except Exception as e:
            logger.error(f"Error getting snapshot for {epic}: {e}")
            return None

class CollaborativeTradingSystem:
    """Implements the three-agent collaborative trading system"""
    
    def __init__(self):
        # Connect to APIs
        self.ig = self._get_ig_service()
        self.polygon = self._get_polygon_client()
        
        # Initialize components
        self.budget = LLMBudgetManager()
        self.memory = TradingMemory()
        self.data = DataCollector(self.ig, self.polygon)
        
        # Set up OpenAI API
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Agent configurations
        self.agents = {
            "scout": {"model": "gpt-3.5-turbo", "cost_estimate": 0.15},
            "strategist": {"model": "gpt-4-turbo-preview", "cost_estimate": 0.50},
            "executor": {"model": "gpt-4-turbo-preview", "cost_estimate": 0.60},
            "team_review": {"model": "gpt-4-turbo-preview", "cost_estimate": 0.40}
        }
        
        # Initialize agent responses
        self.agent_responses = {
            "scout": None,
            "strategist": None,
            "executor": None,
            "team_review": None
        }
    
    def _get_ig_service(self):
        """Connect to IG API"""
        try:
            ig = IGService(
                username=os.getenv("IG_USERNAME"),
                password=os.getenv("IG_PASSWORD"),
                api_key=os.getenv("IG_API_KEY"),
                acc_type=os.getenv("IG_ACC_TYPE", "DEMO")
            )
            ig.create_session()
            logger.info("IG API connected successfully")
            return ig
        except Exception as e:
            logger.error(f"IG connection error: {e}")
            return None
    
    def _get_polygon_client(self):
        """Get Polygon API client"""
        return RESTClient(os.getenv("POLYGON_API_KEY"))
    
    def _call_llm(self, agent, prompt, format_json=True):
        """Call LLM API with appropriate model for each agent"""
        if not self.budget.can_spend(self.agents[agent]["cost_estimate"]):
            logger.warning(f"Insufficient budget for {agent} agent")
            return None
        
        try:
            model = self.agents[agent]["model"]
            
            # Build parameters
            params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": f"You are a forex trading {agent} that works in a collaborative team of trading agents."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3
            }
            
            # Add response format if needed - modified to handle different models
            if format_json and ("turbo" in model or "3.5" in model):
                params["response_format"] = {"type": "json_object"}
            
            # Call API
            response = openai.chat.completions.create(**params)
            
            # Log usage - Error handling improved
            try:
                usage = response.usage
                tokens_in = usage.prompt_tokens
                tokens_out = usage.completion_tokens
                cost = self._calculate_cost(model, tokens_in, tokens_out)
                self.budget.log_usage(agent, tokens_in, tokens_out, cost)
            except Exception as usage_error:
                logger.error(f"Error logging usage: {usage_error}")
                # Log a minimal amount to continue operation
                cost = self.agents[agent]["cost_estimate"] / 2
                self.budget.log_usage(agent, 0, 0, cost)
            
            # Log results
            with open(f"data/{agent}_results.jsonl", "a") as f:
                f.write(json.dumps({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "prompt_tokens": tokens_in if 'tokens_in' in locals() else 0,
                    "completion_tokens": tokens_out if 'tokens_out' in locals() else 0,
                    "result": response.choices[0].message.content
                }) + "\n")
            
            # Parse response
            result = response.choices[0].message.content
            if format_json:
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    # Try to extract JSON if wrapped in code blocks
                    import re
                    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', result)
                    if json_match:
                        try:
                            return json.loads(json_match.group(1))
                        except:
                            logger.error(f"Failed to parse JSON from response for {agent}")
                            return None
                    else:
                        logger.error(f"Failed to extract JSON from response for {agent}")
                        return None
            else:
                return result
        except Exception as e:
            logger.error(f"LLM API error in {agent} agent: {e}")
            return None
    
    def _calculate_cost(self, model, tokens_in, tokens_out):
        """Calculate cost of API call"""
        if "3.5" in model:
            return (tokens_in * 0.0015 + tokens_out * 0.002) / 1000
        elif "gpt-4" in model:
            return (tokens_in * 0.03 + tokens_out * 0.06) / 1000
        else:
            return 0
    
    def execute_trade(self, trade):
        """Execute a new trade"""
        try:
            logger.info(f"Executing {trade.get('direction')} {trade.get('epic')} | Size: {trade.get('size')}")
            
            response = self.ig.create_open_position(
                epic=trade["epic"],
                direction=trade["direction"],
                size=float(trade["size"]),
                order_type="MARKET",
                currency_code=os.getenv("ACCOUNT_CURRENCY", "GBP"),
                expiry="DFB",
                force_open=True,
                guaranteed_stop=False,
                stop_level=float(trade["initial_stop_loss"]) if "initial_stop_loss" in trade else None,
                limit_level=float(trade["take_profit_levels"][0]) if "take_profit_levels" in trade and trade["take_profit_levels"] else None
            )
            
            # Prepare trade log data
            trade_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "epic": trade["epic"],
                "direction": trade["direction"],
                "size": trade["size"],
                "entry_price": trade.get("entry_price"),
                "stop_loss": trade.get("initial_stop_loss"),
                "take_profit": trade.get("take_profit_levels")[0] if trade.get("take_profit_levels") else None,
                "risk_percent": trade.get("risk_percent"),
                "risk_reward": trade.get("risk_reward"),
                "pattern": trade.get("pattern"),
                "stop_management": trade.get("stop_management", []),
                "outcome": "EXECUTED" if response.get("dealStatus") == "ACCEPTED" else "FAILED",
                "deal_id": response.get("dealId"),
                "reason": response.get("reason", "")
            }
            
            # Log the trade
            self.memory.log_trade(trade_data)
            
            # Save analysis for this pair
            if "epic" in trade:
                self.memory.update_analysis_history(trade["epic"], {
                    "direction": trade.get("direction"),
                    "entry_price": trade.get("entry_price"),
                    "stop_loss": trade.get("initial_stop_loss"),
                    "take_profit": trade.get("take_profit_levels"),
                    "risk_reward": trade.get("risk_reward"),
                    "pattern": trade.get("pattern"),
                    "reasoning": trade.get("reasoning")
                })
            
            return True, trade_data
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return False, {"outcome": "ERROR", "reason": str(e)}
    
    def close_position(self, position_action):
        """Close an existing position"""
        try:
            deal_id = position_action.get("dealId")
            epic = position_action.get("epic")
            
            logger.info(f"Closing position {deal_id} | {epic}")
            
            # Find position details
            positions = self.data.get_positions()
            position = positions[positions["dealId"] == deal_id]
            
            if position.empty:
                logger.warning(f"Position not found for close: {deal_id}")
                return False, {"outcome": "FAILED", "reason": "Position not found"}
                
            # Get position details
            direction = position.iloc[0].get("direction")
            size = position.iloc[0].get("size")
            
            # Execute close
            close_direction = "SELL" if direction == "BUY" else "BUY"
            
            response = self.ig.close_open_position(
                deal_id=deal_id,
                direction=close_direction,
                size=float(size),
                order_type="MARKET"
            )
            
            # Log the close
            close_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "epic": epic,
                "direction": "CLOSE",
                "outcome": "CLOSED" if response.get("dealStatus") == "ACCEPTED" else "FAILED",
                "deal_id": deal_id,
                "reason": position_action.get("reason", "")
            }
            
            self.memory.log_trade(close_data)
            
            return True, close_data
        except Exception as e:
            logger.error(f"Close position error: {e}")
            return False, {"outcome": "ERROR", "reason": str(e)}
    
    def update_stop_loss(self, position_action):
        """Update stop loss for an existing position"""
        try:
            deal_id = position_action.get("dealId")
            epic = position_action.get("epic")
            new_level = position_action.get("new_level")
            
            logger.info(f"Updating stop for {deal_id} to {new_level}")
            
            response = self.ig.update_open_position(
                deal_id=deal_id,
                stop_level=float(new_level)
            )
            
            # Log the update
            update_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "epic": epic,
                "action_type": "UPDATE_STOP",
                "deal_id": deal_id,
                "new_level": new_level,
                "outcome": "UPDATED" if response.get("dealStatus") == "ACCEPTED" else "FAILED",
                "reason": position_action.get("reason", "")
            }
            
            # Log as trade for consistency
            self.memory.log_trade(update_data)
            
            return True, update_data
        except Exception as e:
            logger.error(f"Update stop loss error: {e}")
            return False, {"outcome": "ERROR", "reason": str(e)}
    
    def run_scout_agent(self):
        """Run the Market Scout agent to identify opportunities"""
        logger.info("Running Scout Agent")
        
        try:
            # Collect context data
            account_data = self.data.get_account_data()
            positions = self.data.get_positions()
            recent_trades = self.memory.get_recent_trades(5)
            all_trades = self.memory.get_all_trades()
            
            # Collect market data for all pairs
            market_data = {}
            for epic in FOREX_PAIRS:
                data = self.data.get_market_data(epic)
                if data:
                    market_data[epic] = data
            
            # Get agent feedback
            agent_feedback = self.memory.get_agent_feedback()
            
            # Build prompt using template
            scout_prompt = CollaborativeTradingPrompts.market_scanner(
                market_data, 
                account_data, 
                positions, 
                recent_trades, 
                all_trades, 
                agent_feedback
            )
            
            # Call LLM
            scout_result = self._call_llm("scout", scout_prompt)
            
            if scout_result:
                logger.info(f"Scout found {len(scout_result.get('opportunities', []))} opportunities")
                self.agent_responses["scout"] = scout_result
                
                # Save self-improvement suggestions for team review
                if "self_improvement" in scout_result:
                    self.memory.update_feedback("scout", scout_result["self_improvement"])
                
                return scout_result
            else:
                logger.warning("Scout agent produced no result")
                return None
            
        except Exception as e:
            logger.error(f"Error in scout agent: {e}")
            return None
    
    def run_strategist_agent(self, scout_result):
        """Run the Strategic Analyst agent to analyze opportunities"""
        logger.info("Running Strategist Agent")
        
        try:
            # Exit if no scout result or opportunities
            if not scout_result or "opportunities" not in scout_result:
                logger.warning("No opportunities from scout to analyze")
                return None
            
            # Get opportunities from scout
            opportunities = scout_result.get("opportunities", [])
            
            # Exit if no opportunities
            if not opportunities:
                logger.warning("No opportunities from scout to analyze")
                return None
            
            # Collect context data
            account_data = self.data.get_account_data()
            positions = self.data.get_positions()
            
            # Collect market data for selected pairs
            market_data = {}
            for opp in opportunities:
                epic = opp.get("epic")
                data = self.data.get_market_data(epic)
                if data:
                    market_data[epic] = data
            
            # Get previous analyses
            previous_analyses = self.memory.get_pair_analysis_history()
            
            # Build prompt using template
            strategist_prompt = CollaborativeTradingPrompts.analysis_engine(
                opportunities,
                market_data,
                account_data,
                positions,
                self.memory.memory,
                previous_analyses
            )
            
            # Call LLM
            strategist_result = self._call_llm("strategist", strategist_prompt)
            
            if strategist_result:
                logger.info(f"Strategist analyzed {len(strategist_result.get('analysis_results', []))} pairs")
                self.agent_responses["strategist"] = strategist_result
                
                # Save self-improvement suggestions for team review
                if "self_improvement" in strategist_result:
                    self.memory.update_feedback("strategist", strategist_result["self_improvement"])
                
                return strategist_result
            else:
                logger.warning("Strategist agent produced no result")
                return None
            
        except Exception as e:
            logger.error(f"Error in strategist agent: {e}")
            return None
    
    def run_executor_agent(self, strategist_result):
        """Run the Decision Executor agent to make trading decisions"""
        logger.info("Running Executor Agent")
        
        try:
            # Exit if no strategist result or analysis
            if not strategist_result or "analysis_results" not in strategist_result:
                logger.warning("No analysis from strategist for execution")
                return None
            
            # Get analysis from strategist
            analysis_results = strategist_result.get("analysis_results", [])
            
            # Exit if no analysis
            if not analysis_results:
                logger.warning("No analysis from strategist for execution")
                return None
            
            # Collect context data
            account_data = self.data.get_account_data()
            positions = self.data.get_positions()
            recent_trades = self.memory.get_recent_trades(5)
            all_trades = self.memory.get_all_trades()
            
            # Collect market data for all pairs
            market_data = {}
            for epic in FOREX_PAIRS:
                data = self.data.get_market_data(epic)
                if data:
                    market_data[epic] = data
            
            # Build prompt using template
            executor_prompt = CollaborativeTradingPrompts.decision_maker(
                analysis_results,
                account_data,
                positions,
                self.memory.memory,
                market_data,
                recent_trades,
                all_trades
            )
            
            # Call LLM
            executor_result = self._call_llm("executor", executor_prompt)
            
            if executor_result:
                logger.info(f"Executor generated {len(executor_result.get('trade_actions', []))} trades and {len(executor_result.get('position_actions', []))} position actions")
                self.agent_responses["executor"] = executor_result
                
                # Save self-improvement suggestions for team review
                if "self_improvement" in executor_result:
                    self.memory.update_feedback("executor", executor_result["self_improvement"])
                
                return executor_result
            else:
                logger.warning("Executor agent produced no result")
                return None
            
        except Exception as e:
            logger.error(f"Error in executor agent: {e}")
            return None
    
    def run_team_review(self):
        """Run team review to coordinate and improve the agents"""
        logger.info("Running Team Review")
        
        try:
            # Skip if not all agents have run
            if not self.agent_responses["scout"] or not self.agent_responses["strategist"] or not self.agent_responses["executor"]:
                logger.warning("Cannot run team review - missing agent responses")
                return None
            
            # Collect context data
            account_data = self.data.get_account_data()
            positions = self.data.get_positions()
            
            # Collect market data for basic context
            market_data = {}
            for epic in FOREX_PAIRS[:5]:  # Just get a few for context
                data = self.data.get_market_data(epic)
                if data:
                    market_data[epic] = data
            
            # Calculate daily performance
            # In a production system, this would come from actual P&L tracking
            daily_perf = {
                "profit_loss": 0,  # Placeholder
                "return_percent": self.memory.memory.get("daily_return", 0),
                "winning_trades": 0,  # Placeholder
                "losing_trades": 0    # Placeholder
            }
            
            # Build prompt using template
            team_review_prompt = CollaborativeTradingPrompts.team_review(
                self.agent_responses,
                self.memory.memory,
                market_data,
                positions,
                account_data,
                daily_perf
            )
            
            # Call LLM
            team_result = self._call_llm("team_review", team_review_prompt)
            
            if team_result:
                logger.info("Team review completed")
                self.agent_responses["team_review"] = team_result
                
                # Update agent feedback from team review
                if "agent_feedback" in team_result:
                    for agent, feedback in team_result["agent_feedback"].items():
                        self.memory.update_feedback(agent, feedback)
                
                # Log any requests for the human operator
                if "requests_for_human" in team_result:
                    requests = team_result["requests_for_human"]
                    logger.info(f"Requests for human operator: {requests}")
                    
                    # Save to a dedicated file for the human to review
                    with open("data/human_requests.jsonl", "a") as f:
                        f.write(json.dumps({
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "requests": requests
                        }) + "\n")
                
                return team_result
            else:
                logger.warning("Team review produced no result")
                return None
            
        except Exception as e:
            logger.error(f"Error in team review: {e}")
            return None
    
    def execute_trading_actions(self, executor_result):
        """Execute the trading actions recommended by the executor agent"""
        logger.info("Executing trading actions")
        
        try:
            # Skip if no executor result
            if not executor_result:
                logger.warning("No executor result to implement")
                return False
            
            # Execute new trades
            trade_actions = executor_result.get("trade_actions", [])
            for trade in trade_actions:
                if trade.get("action_type") == "OPEN":
                    success, trade_result = self.execute_trade(trade)
                    if success:
                        logger.info(f"Successfully executed trade: {trade.get('epic')} {trade.get('direction')}")
                    else:
                        logger.error(f"Failed to execute trade: {trade_result}")
            
            # Execute position actions
            position_actions = executor_result.get("position_actions", [])
            for action in position_actions:
                action_type = action.get("action_type", "").upper()
                
                if action_type == "CLOSE":
                    success, result = self.close_position(action)
                    if success:
                        logger.info(f"Successfully closed position: {action.get('epic')} {action.get('dealId')}")
                    else:
                        logger.error(f"Failed to close position: {result}")
                        
                elif action_type == "UPDATE_STOP":
                    success, result = self.update_stop_loss(action)
                    if success:
                        logger.info(f"Successfully updated stop: {action.get('epic')} {action.get('dealId')} to {action.get('new_level')}")
                    else:
                        logger.error(f"Failed to update stop: {result}")
            
            # Update daily return tracking (placeholder - would use actual P&L in production)
            # This is just a simple counter for demonstration purposes
            current_return = self.memory.memory.get("daily_return", 0)
            trade_count = len(trade_actions) + len(position_actions)
            if trade_count > 0:
                # Simulate some progress toward 10% goal
                self.memory.update_memory("daily_return", min(10.0, current_return + (0.5 * trade_count)))
            
            return True
        except Exception as e:
            logger.error(f"Error executing trading actions: {e}")
            return False
    
    def run_trading_cycle(self):
        """Run a complete trading cycle with all three agents"""
        logger.info("Starting collaborative trading cycle")
        
        try:
            # Reset agent responses for this cycle
            self.agent_responses = {
                "scout": None,
                "strategist": None,
                "executor": None,
                "team_review": None
            }
            
            # 1. Run Market Scout Agent
            scout_result = self.run_scout_agent()
            
            # 2. Run Strategist Agent if scout found opportunities
            if scout_result and scout_result.get("opportunities"):
                strategist_result = self.run_strategist_agent(scout_result)
                
                # 3. Run Executor Agent if strategist produced analysis
                if strategist_result and strategist_result.get("analysis_results"):
                    executor_result = self.run_executor_agent(strategist_result)
                    
                    # 4. Execute trading actions
                    if executor_result:
                        self.execute_trading_actions(executor_result)
            
            # 5. Run team review if all agents produced results
            if all(self.agent_responses.values()):
                self.run_team_review()
            
            # Log budget status
            budget_status = self.budget.get_status()
            logger.info(f"Budget status: ${budget_status['remaining']:.2f} remaining ({100-budget_status['percent_used']:.1f}%)")
            
            return True
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            return False
    
    def run(self):
        """Main trading loop"""
        logger.info("Starting Collaborative LLM Forex Trading System")
        print("\nSTARTING COLLABORATIVE LLM FOREX TRADING SYSTEM")
        
        # Display initial budget
        budget_status = self.budget.get_status()
        print(f"Daily budget: ${budget_status['total_budget']:.2f}")
        print(f"Available: ${budget_status['remaining']:.2f}")
        
        # Trading loop
        while True:
            try:
                # Run a full trading cycle
                self.run_trading_cycle()
                
                # Sleep between cycles - adjust based on market activity
                # Market hours could influence the sleep duration
                current_hour = datetime.now(timezone.utc).hour
                
                # More frequent during active market hours
                if 8 <= current_hour <= 16:  # Major market hours (approx)
                    sleep_time = 5 * 60  # 5 minutes
                else:
                    sleep_time = 15 * 60  # 15 minutes
                
                logger.info(f"Cycle complete. Sleeping for {sleep_time/60:.1f} minutes until next cycle.")
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait 1 minute on error

# --- Main Entry Point ---
if __name__ == "__main__":
    trading_system = CollaborativeTradingSystem()
    trading_system.run()