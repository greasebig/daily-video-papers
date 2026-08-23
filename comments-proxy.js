#!/usr/bin/env node
'use strict';
/*
 * Guestbook comment proxy — listens on port 7897.
 * The browser widget POSTs here; this server adds the GitHub token and
 * appends the comment to comments/data.json in the repo (file-backed store,
 * the same way the paper site commits its content). The token NEVER reaches
 * the browser, so the page never redirects to GitHub.
 *
 * Usage:
 *   GITHUB_TOKEN=ghp_xxx GITHUB_REPO=greasebig/greasebig.github.io node comments-proxy.js
 *
 * Env:
 *   PORT            (default 7897)
 *   GITHUB_TOKEN    (required) a PAT with repo contents write access
 *   GITHUB_REPO     (default greasebig/greasebig.github.io) owner/repo
 *   GITHUB_DATA     (default comments/data.json) path to the comment store
 *   ALLOW_ORIGIN    (default *) CORS allowed origin for the comment page
 */
const http = require('http');
const https = require('https');

const PORT = process.env.PORT || 7897;
const TOKEN = process.env.GITHUB_TOKEN || '';
const REPO = process.env.GITHUB_REPO || 'greasebig/greasebig.github.io';
const DATA_PATH = process.env.GITHUB_DATA || 'comments/data.json';
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

// GET the current data file (returns {arr, sha} or {arr:[], sha:null} if 404).
function readData(cb) {
  const req = https.request({
    hostname: API,
    path: '/repos/' + REPO + '/contents/' + DATA_PATH,
    method: 'GET',
    headers: {
      'Authorization': 'Bearer ' + TOKEN,
      'Accept': 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
      'User-Agent': 'gh-comments-proxy'
    }
  }, function (r) {
    let data = '';
    r.on('data', function (c) { data += c; });
    r.on('end', function () {
      if (r.statusCode === 200) {
        let info; try { info = JSON.parse(data); } catch (e) { info = null; }
        let arr = [];
        if (info && info.content) {
          try { arr = JSON.parse(Buffer.from(info.content, 'base64').toString('utf-8')); } catch (e) { arr = []; }
        }
        cb(200, arr, info ? info.sha : null);
      } else if (r.statusCode === 404) {
        cb(404, [], null);
      } else {
        let json; try { json = JSON.parse(data); } catch (e) { json = {}; }
        cb(r.statusCode, null, null, json);
      }
    });
  });
  req.on('error', function (e) { cb(0, null, null, { error: e.message }); });
  req.end();
}

// PUT the updated data file back (sha required unless creating).
function writeData(arr, sha, cb) {
  const content = Buffer.from(JSON.stringify(arr, null, 2), 'utf-8').toString('base64');
  const payload = sha
    ? JSON.stringify({ message: 'guestbook: add comment', content: content, sha: sha })
    : JSON.stringify({ message: 'guestbook: add comment', content: content });
  const req = https.request({
    hostname: API,
    path: '/repos/' + REPO + '/contents/' + DATA_PATH,
    method: 'PUT',
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

function postComment(name, body, cb) {
  readData(function (code, arr, sha, errJson) {
    if (code !== 200 && code !== 404) { cb(code, errJson || {}); return; }
    if (!Array.isArray(arr)) arr = [];
    arr.push({
      user: { login: name && name.length ? name : 'guest' },
      created_at: new Date().toISOString(),
      body: body
    });
    writeData(arr, sha, function (wcode, wjson) {
      if (wcode >= 200 && wcode < 300) cb(200, { ok: true });
      else cb(wcode || 502, wjson);
    });
  });
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
      const name = (p.name || '').toString().trim().slice(0, 40);   // optional; no nickname UI
      const body = (p.body || '').toString().trim().slice(0, 2000);
      if (!body) { send(res, 400, { error: 'body required' }, origin); return; }
      postComment(name, body, function (code, json) {
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
