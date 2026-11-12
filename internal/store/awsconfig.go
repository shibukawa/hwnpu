package store

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"

	"github.com/shibukawa/tinygodriver/cloud/aws"
)

// profile is what hwnpu reads of a profile of the AWS CLI's files: the static keys of
// ~/.aws/credentials (AWS_SHARED_CREDENTIALS_FILE) and the region and the endpoint of
// ~/.aws/config (AWS_CONFIG_FILE). SSO, assumed roles and the instance's metadata are not read
// (docs/cache-sharing.md 7.2).
type profile struct {
	creds            aws.Credentials
	region, endpoint string
}

// readProfile reads the profile name; "" is AWS_PROFILE, else the environment's keys, else
// default's.
func readProfile(name string) profile {
	var p profile
	if name == "" {
		name = os.Getenv("AWS_PROFILE")
	}
	if name == "" {
		if c := aws.CredentialsFromEnv(); c.Valid() {
			p.creds = c
			return p
		}
		name = "default"
	}
	home, _ := os.UserHomeDir()
	credFile := firstOf(os.Getenv("AWS_SHARED_CREDENTIALS_FILE"), filepath.Join(home, ".aws", "credentials"))
	confFile := firstOf(os.Getenv("AWS_CONFIG_FILE"), filepath.Join(home, ".aws", "config"))
	cred := iniSection(credFile, name)
	conf := iniSection(confFile, "profile "+name)
	if name == "default" && len(conf) == 0 {
		conf = iniSection(confFile, "default")
	}
	get := func(k string) string { return firstOf(cred[k], conf[k]) }
	p.creds = aws.Credentials{AccessKeyID: get("aws_access_key_id"), SecretAccessKey: get("aws_secret_access_key"), SessionToken: get("aws_session_token")}
	p.region = conf["region"]
	p.endpoint = firstOf(conf["endpoint_url"], conf["s3.endpoint_url"])
	return p
}

// iniSection reads the keys of section [name] of an INI file; the nested settings of the AWS
// CLI's config (s3 =\n  endpoint_url = ...) are read as s3.endpoint_url.
func iniSection(file, name string) map[string]string {
	out := map[string]string{}
	f, err := os.Open(file)
	if err != nil {
		return out
	}
	defer f.Close()
	in, nest := false, ""
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		raw := sc.Text()
		line := strings.TrimSpace(raw)
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, ";") {
			continue
		}
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			in = strings.TrimSpace(line[1:len(line)-1]) == name
			nest = ""
			continue
		}
		if !in {
			continue
		}
		k, v, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		k, v = strings.TrimSpace(k), strings.TrimSpace(v)
		indented := raw != "" && (raw[0] == ' ' || raw[0] == '\t')
		switch {
		case v == "":
			nest = k
		case indented && nest != "":
			out[nest+"."+k] = v
		default:
			nest = ""
			out[k] = v
		}
	}
	return out
}
