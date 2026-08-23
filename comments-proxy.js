#!/usr/bin/env node
'use strict';
/*
 * GitHub-Issues comment proxy — listens on port 7897.
 * The browser widget POSTs here; this server adds the GitHub token
 * and forwards the comment to the issue via the GitHub API.
 * The token NEVER reaches the browser, so the page never redirects to GitHub.
 *
 * Usage:
 *   GITHUB_TOKEN=ghp_xxx GITHUB_REPO=greasebig/daily-video-papers node comments-proxy.js
 *
 * Env:
 *   PORT          (default 7897)
 *   GITHUB_TOKEN  (required) a PAT with 'repo' or 'public_repo' scope
 *   GITHUB_REPO   (default greasebig/daily-video-papers) owner/repo to post into
 *   ALLOW_ORIGIN  (default *) CORS allowed origin for the comment page
 */
const http = require('http');
const https = require('https');

const PORT = process.env.PORT || 7897;
const TOKEN = process.env.GITHUB_TOKEN || '';
const REPO = process.env.GITHUB_REPO || 'greasebig/greasebig.github.io';
const ALLOW_ORIGIN = process.env.ALLOW_ORIGIN || '*';
const API = 'api.github.com';

function send(res, code, obj, origin) {
  res.writeHead(code, {
    'Content-Type': 'application/json; charset=utf-8',
    'Access-Control-Allow-Origin': origin || ALLOW_ORIGIN,
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type'
  });
  res.end(JSON.stringify(obj));
}

function postComment(issue, name, body, cb) {
  const stamp = (name && name.length) ? ('**' + name + '** 留言于 ' + new Date().toISOString() + '：\n\n') : '';
  const payload = JSON.stringify({ body: stamp + body });
  const req = https.request({
    hostname: API,
    path: '/repos/' + REPO + '/issues/' + issue + '/comments',
    method: 'POST',
    headers: {
      'Authorization': 'Bearer ' + TOKEN,
      'Accept': 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
      'User-Agent': 'gh-comments-proxy',
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(payload)
    }
  }, function (r) {
    let data = '';
    r.on('data', function (c) { data += c; });
    r.on('end', function () {
      let json; try { json = JSON.parse(data); } catch (e) { json = {}; }
      cb(r.statusCode, json);
    });
  });
  req.on('error', function (e) { cb(0, { error: e.message }); });
  req.write(payload);
  req.end();
}

const server = http.createServer(function (req, res) {
  const origin = req.headers.origin || '*';
  if (req.method === 'OPTIONS') { send(res, 204, {}, origin); return; }
  if (req.url === '/health') { send(res, 200, { ok: true, repo: REPO }); return; }

  if (req.method === 'POST' && req.url === '/api/comment') {
    if (!TOKEN) { send(res, 500, { error: 'GITHUB_TOKEN not set' }, origin); return; }
    let raw = '';
    req.on('data', function (c) { raw += c; if (raw.length > 1e6) req.destroy(); });
    req.on('end', function () {
      let p; try { p = JSON.parse(raw); } catch (e) { send(res, 400, { error: 'bad json' }, origin); return; }
      const issue = parseInt(p.issue, 10);
      const name = (p.name || '').toString().trim().slice(0, 40);
      const body = (p.body || '').toString().trim().slice(0, 2000);
      if (!issue || issue <= 0) { send(res, 400, { error: 'invalid issue' }, origin); return; }
      if (!body) { send(res, 400, { error: 'body required' }, origin); return; }
      postComment(issue, name, body, function (code, json) {
        if (code >= 200 && code < 300) send(res, 200, { ok: true, comment: json }, origin);
        else send(res, code || 502, { error: (json && json.message) || 'github error', detail: json }, origin);
      });
    });
    return;
  }
  send(res, 404, { error: 'not found' }, origin);
});

server.listen(PORT, function () {
  console.log('Comments proxy listening on http://localhost:' + PORT);
  console.log('Target repo: ' + REPO + (TOKEN ? '  (token: set)' : '  (token: MISSING)'));
});
